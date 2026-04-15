import ast
import glob
import logging
import os
from datetime import datetime
import torch
import multiprocessing as mp
import polars as pl
import ase.io
from ase.optimize import LBFGS
from fairchem.data.oc.core import Slab
from fairchem.core.components.calculate.recipes.adsorbml import run_adsorbml
from fairchem.core import FAIRChemCalculator

# ── Config ────────────────────────────────────────────────────────────────────
ADSORBATE_SMILES = "*H"
SLABS_TO_VERIFY  = "adsorbml_prepared.csv"
OUT_DIR          = "adsorbml_results/uma-m_run-1"
MIN_FREE_VRAM_GB = 8.0   # GPUs with less free VRAM are skipped
WORKERS_PER_GPU  = 1     # raise to 2 on 80 GB A100s; keep 1 for 24–40 GB GPUs

# Master logger: used by the orchestration layer (__main__, detect_gpus, worker).
master_log = logging.getLogger("adsorbml.master")

_LOG_FMT = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
# Compound-level formatter includes the logger name so concurrent workers are
# identifiable when their console output is interleaved.
_COMPOUND_LOG_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logging(log_path: str | None = None) -> None:
    """Configure the master logger with a console handler and optional file handler.

    Called by the main process (with log_path for the run-level log) and by
    each worker subprocess (with log_path for the per-worker log).
    Per-compound file logging is handled inside process_row.
    """
    master_log.setLevel(logging.DEBUG)
    if master_log.handlers:          # avoid duplicate handlers if called twice
        return
    console = logging.StreamHandler()
    console.setFormatter(_LOG_FMT)
    master_log.addHandler(console)

    if log_path:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(_LOG_FMT)
        master_log.addHandler(fh)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_tuple(x):
    """Parse a millers string like '(1, 1, 0)' into a Python tuple; return None on failure."""
    if isinstance(x, tuple):
        return x
    try:
        v = ast.literal_eval(str(x))
        return v if isinstance(v, tuple) else None
    except Exception:
        return None


def detect_gpus(min_free_gb=MIN_FREE_VRAM_GB):
    """Return list of eligible GPU indices ordered by free VRAM (most first)."""
    if not torch.cuda.is_available():
        return []
    eligible = []
    master_log.info("GPU inventory:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        free_gb  = free_bytes  / 1e9
        total_gb = total_bytes / 1e9
        ok = free_gb >= min_free_gb
        status = "OK" if ok else "LOW VRAM – skipped"
        master_log.info(f"  GPU {i}: {props.name}  {free_gb:.1f}/{total_gb:.1f} GB free  [{status}]")
        if ok:
            eligible.append((free_gb, i))
    eligible.sort(reverse=True)
    return [i for _, i in eligible]


def _is_done(row):
    run_label = os.path.splitext(os.path.basename(row["slab_file"]))[0]
    return os.path.exists(os.path.join(OUT_DIR, run_label, "candidates.csv"))


# ── Per-row processing (runs inside worker process) ───────────────────────────
_EMPTY_SCHEMA = {
    "slab_name": pl.Utf8, "slab_id": pl.Utf8, "millers": pl.Utf8,
    "termination_idx": pl.Int64, "dopant": pl.Utf8,
    "vacancy_element": pl.Utf8, "vacancy_count": pl.Int64,
    "cluster_element": pl.Utf8, "cluster_count": pl.Int64,
    "candidate_rank": pl.Int64,
    "E_adslab_ml_eV": pl.Float64, "E_slab_ml_eV": pl.Float64,
    "E_gas_ref_ml_eV": pl.Float64, "E_ads_ml_eV": pl.Float64,
    "anomalies": pl.Utf8, "traj_path": pl.Utf8,
}


def _close_compound_log(comp_log: logging.Logger) -> None:
    """Close and remove all handlers from a per-compound logger.

    Prevents file-descriptor and handler accumulation when a worker processes
    many compounds sequentially.
    """
    for h in comp_log.handlers[:]:
        h.close()
        comp_log.removeHandler(h)


def process_row(row, calc):
    """Run AdsorbML for a single slab row and write candidates.csv to its run_dir."""
    slab_file = row["slab_file"]
    slab_name = row["slab_name"]
    millers   = row["millers"]
    term_idx  = row["termination_idx"]

    run_label = os.path.splitext(os.path.basename(slab_file))[0]
    run_dir   = os.path.join(OUT_DIR, run_label)
    done_csv  = os.path.join(run_dir, "candidates.csv")
    os.makedirs(run_dir, exist_ok=True)

    # Per-compound logger: writes to run_dir/adsorbml.log.
    # propagate=False keeps compound messages out of the root logger.
    comp_log = logging.getLogger(f"adsorbml.{run_label}")
    comp_log.setLevel(logging.DEBUG)
    comp_log.propagate = False
    if not comp_log.handlers:
        fh = logging.FileHandler(
            os.path.join(run_dir, "adsorbml.log"), mode="a", encoding="utf-8"
        )
        fh.setFormatter(_LOG_FMT)
        comp_log.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(_COMPOUND_LOG_FMT)
        comp_log.addHandler(ch)

    # Race-condition guard: another worker may have finished this between the
    # main-process resume filter and now.
    if os.path.exists(done_csv):
        comp_log.info(f"SKIP (already done): {run_label}")
        _close_compound_log(comp_log)
        return

    comp_log.info("=" * 50)
    comp_log.info(f"Running: {run_label}")

    try:
        atoms = ase.io.read(slab_file)
        slab  = Slab(bulk=None, slab_atoms=atoms, millers=millers,
                     shift=None, top=None, oriented_bulk=None)
    except Exception as e:
        comp_log.error(f"Could not load slab: {e}")
        _close_compound_log(comp_log)
        return

    try:
        outputs = run_adsorbml(
            slab=slab,
            adsorbate=ADSORBATE_SMILES,
            calculator=calc,
            optimizer_cls=LBFGS,
            fmax=0.02,
            steps=200,
            num_placements=100,
            reference_ml_energies=True,
            place_on_relaxed_slab=False,
        )
    except Exception as e:
        comp_log.error(f"run_adsorbml failed: {e}")
        _close_compound_log(comp_log)
        return

    candidates = outputs["adslabs"]

    if not candidates:
        comp_log.warning(f"No valid placements for {run_label}")
        # Write empty-schema CSV so resume skips this row next time
        pl.DataFrame(schema=_EMPTY_SCHEMA).write_csv(done_csv)
        _close_compound_log(comp_log)
        return

    for i, cand in enumerate(candidates):
        ase.io.write(os.path.join(run_dir, f"candidate_{i}.traj"), cand["atoms"])

    cand_rows = []
    for i, cand in enumerate(candidates):
        res = cand["results"]
        ref = res.get("referenced_adsorption_energy", {})
        cand_rows.append(dict(
            slab_name       = slab_name,
            slab_id         = row["slab_id"],
            millers         = str(millers),
            termination_idx = term_idx,
            dopant          = row["dopant"],
            vacancy_element = row["vacancy_element"],
            vacancy_count   = row["vacancy_count"],
            cluster_element = row["cluster_element"],
            cluster_count   = row["cluster_count"],
            candidate_rank  = i,
            E_adslab_ml_eV  = res.get("energy", float("nan")),
            E_slab_ml_eV    = ref.get("slab_energy", float("nan")),
            E_gas_ref_ml_eV = ref.get("gas_reactant_energy", float("nan")),
            E_ads_ml_eV     = ref.get("adsorption_energy", float("nan")),
            anomalies       = "|".join(res.get("adslab_anomalies", [])),
            traj_path       = os.path.join(run_dir, f"candidate_{i}.traj"),
        ))

    cand_df = pl.DataFrame(cand_rows)
    cand_df.write_csv(done_csv)

    best_e = cand_df["E_ads_ml_eV"].min()
    comp_log.info(f"Best E_ads (ML) = {best_e:.4f} eV  ({len(candidates)} candidates)")
    _close_compound_log(comp_log)


# ── Worker process entry point ────────────────────────────────────────────────
def worker(gpu_id, worker_idx, task_queue, out_dir):
    # Each worker gets its own master log file for orchestration-level events
    # (calculator load, sentinels).  Per-compound events go to run_dir/adsorbml.log.
    log_path = os.path.join(out_dir, f"worker_gpu{gpu_id}_w{worker_idx}.log")
    setup_logging(log_path)

    if gpu_id is not None:
        # FAIRChemCalculator only accepts "cuda", not "cuda:N".
        # Pin this process to the right physical GPU via the environment variable.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda"
    else:
        device = "cpu"
    master_log.info(f"Loading calculator on {device} (GPU {gpu_id})...")
    calc = FAIRChemCalculator.from_model_checkpoint("uma-m-1p1", task_name="oc20", device=device)
    while True:
        row = task_queue.get()
        if row is None:   # sentinel — no more work
            break
        process_row(row, calc)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    log_path = os.path.join(OUT_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_path)
    master_log.info(f"Log file: {log_path}")

    # Detect GPUs
    master_log.info("Detecting GPUs...")
    gpu_ids   = detect_gpus()
    n_workers = len(gpu_ids) * WORKERS_PER_GPU if gpu_ids else 1
    if gpu_ids:
        master_log.info(f"Launching {n_workers} worker(s) across GPU(s): {gpu_ids}")
    else:
        master_log.warning("No eligible GPU found — running on CPU (slow).")

    # Load manifest
    df = pl.read_csv(SLABS_TO_VERIFY)
    df = df.with_columns(pl.col("millers").map_elements(_to_tuple, return_dtype=pl.Object))

    # Resume: skip already-done rows
    all_rows     = list(df.iter_rows(named=True))
    pending_rows = [r for r in all_rows if not _is_done(r)]
    done_count   = len(all_rows) - len(pending_rows)
    master_log.info(f"Total: {len(all_rows)}  |  Done: {done_count}  |  Pending: {len(pending_rows)}")

    if not gpu_ids:
        gpu_ids = [None]   # worker already handles gpu_id=None → runs on CPU

    if pending_rows:
        ctx        = mp.get_context("spawn")   # CUDA requires spawn, not fork
        task_queue = ctx.Queue()

        for row in pending_rows:
            task_queue.put(row)
        for _ in range(n_workers):              # one sentinel per worker
            task_queue.put(None)

        processes = [
            ctx.Process(target=worker,
                        args=(gpu_id, i * WORKERS_PER_GPU + j, task_queue, OUT_DIR))
            for i, gpu_id in enumerate(gpu_ids)
            for j in range(WORKERS_PER_GPU)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    # Consolidated summary (glob all candidates.csv written this run or prior)
    all_csvs = sorted(glob.glob(os.path.join(OUT_DIR, "*", "candidates.csv")))
    frames = []
    for csv_path in all_csvs:
        try:
            part = pl.read_csv(csv_path)
            if len(part) > 0:
                frames.append(part)
        except Exception as e:
            master_log.warning(f"Skipping {csv_path}: {e}")

    if frames:
        summary_df   = pl.concat(frames)
        summary_path = os.path.join(OUT_DIR, "adsorbml_batch_summary.csv")
        summary_df.write_csv(summary_path)
        master_log.info(f"Saved consolidated results → {summary_path}")

        group_cols = ["slab_name", "millers", "termination_idx",
                      "dopant", "vacancy_element", "cluster_element"]
        best = (
            summary_df
            .filter(
                pl.col("E_ads_ml_eV")
                == pl.col("E_ads_ml_eV").min().over(group_cols)
            )
            .select(group_cols + ["E_ads_ml_eV"])
        )
        master_log.info("Best ML adsorption energies per system:")
        master_log.info(f"\n{best}")
    else:
        master_log.info("No results to summarise yet.")
