"""
Run GPAW DFT single-point validation on the best AdsorbML candidate from a given run folder.

Computes E_ads = E(adslab) - E(slab) - ½E(H₂), where ½E(H₂) = -3.477 eV is the
PBE reference value from the AdsorbML paper (no H₂ DFT calculation performed).

The best candidate is selected by gibbs_free_ML = E_ads_ml_eV + 0.24 eV closest to 0.

Usage:
    python master_script.py adsorbml_batch/Mo2C_110_t0
    python master_script.py adsorbml_batch/Mo2C_110_t0 --ncores 8
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import textwrap

import polars as pl

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ── Resources ────────────────────────────────────────────────────────────────
# 16 cores × ~8 GB/core ≈ 128 GB RAM ceiling
DEFAULT_NCORES = 4

# ── Reference energy ─────────────────────────────────────────────────────────
# ½E(H₂), PBE reference from the AdsorbML paper — no H₂ DFT calculation needed
E_H2_HALF_REF = -3.477  # eV

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("master_script.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────
def run_gpaw_script(script: str, label: str, run_dir: str, ncores: int) -> dict:
    """Write script to run_dir, execute with mpiexec, return parsed JSON result."""
    script_path = os.path.join(run_dir, f"{label}_gpaw.py")
    result_path = os.path.join(run_dir, f"{label}_result.json")

    with open(script_path, "w") as f:
        f.write(script)

    proc = subprocess.run(
        ["mpiexec", "-n", str(ncores), sys.executable, script_path],
        capture_output=True,
        text=True,
        cwd=run_dir,
    )

    if proc.returncode != 0:
        log.error("GPAW failed for %s:\n%s", label, proc.stderr)
        raise RuntimeError(f"GPAW failed for {label}:\n{proc.stderr}")

    with open(result_path) as f:
        return json.load(f)


# ── Main ─────────────────────────────────────────────────────────────────────
def main(run_dir: str, ncores: int) -> None:
    log.info("Run folder        : %s", run_dir)
    log.info("MPI cores         : %d", ncores)
    log.info("½E(H₂) reference  : %.4f eV (AdsorbML paper, PBE)", E_H2_HALF_REF)

    candidates_path = os.path.join(run_dir, "candidates.csv")
    positions_path  = os.path.join(run_dir, "positions.csv")

    if not os.path.exists(candidates_path):
        raise FileNotFoundError(f"candidates.csv not found in {run_dir}")
    if not os.path.exists(positions_path):
        raise FileNotFoundError(
            f"positions.csv not found in {run_dir}. "
            "Run extract_positions.py first."
        )

    # ── Pick best candidate ───────────────────────────────────────────────────
    cand_df = pl.read_csv(candidates_path)
    cand_df = cand_df.with_columns((pl.col("E_ads_ml_eV") + 0.24).alias("gibbs_free_ML"))
    best_row = cand_df.filter(pl.col("gibbs_free_ML").abs() == pl.col("gibbs_free_ML").abs().min()).row(0, named=True)
    best_rank = int(best_row["candidate_rank"])
    log.info(
        "Best candidate: rank=%d  E_ads_ML=%.4f eV  gibbs_free_ML=%.4f eV (closest to 0)",
        best_rank, best_row["E_ads_ml_eV"], best_row["gibbs_free_ML"],
    )

    # ── Log adsorbate position from positions.csv ─────────────────────────────
    pos_df = pl.read_csv(positions_path)
    best_pos = pos_df.filter(pl.col("candidate_rank") == best_rank).row(0, named=True)
    log.info(
        "H position (ML-relaxed): x=%.4f  y=%.4f  z=%.4f Å",
        best_pos["x"], best_pos["y"], best_pos["z"],
    )

    # Traj files live in run_dir; GPAW scripts run with cwd=run_dir
    slab_traj      = "slab_ml-relaxed.traj"
    candidate_traj = f"candidate_{best_rank}.traj"

    # ── 1. Clean slab (single-point) ─────────────────────────────────────────
    slab_result_path = os.path.join(run_dir, "slab_result.json")
    if os.path.exists(slab_result_path):
        with open(slab_result_path) as f:
            e_slab = json.load(f)["energy"]
        log.info("Slab result cached — skipping. E(slab) = %.4f eV", e_slab)
    else:
        slab_script = textwrap.dedent(f"""\
            import json, ase.io
            from gpaw import GPAW, PW

            atoms = ase.io.read('{slab_traj}')
            atoms.calc = GPAW(
                mode=PW(400),
                xc='PBE',
                kpts={{'size': (4, 4, 1), 'gamma': True}},
                occupations={{'name': 'fermi-dirac', 'width': 0.1}},
                convergence={{'energy': 1e-5}},
                txt='slab.txt',
            )
            e = atoms.get_potential_energy()
            with open('slab_result.json', 'w') as f:
                json.dump({{'energy': e}}, f)
        """)
        log.info("Running clean slab single-point DFT...")
        e_slab = run_gpaw_script(slab_script, "slab", run_dir, ncores)["energy"]
    log.info("E(slab) = %.4f eV", e_slab)

    # ── 2. Best adsorbate+slab candidate (single-point) ──────────────────────
    adslab_result_path = os.path.join(run_dir, "adslab_result.json")
    if os.path.exists(adslab_result_path):
        with open(adslab_result_path) as f:
            e_adslab = json.load(f)["energy"]
        log.info("Adslab result cached — skipping. E(adslab) = %.4f eV", e_adslab)
    else:
        adslab_script = textwrap.dedent(f"""\
            import json, ase.io
            from gpaw import GPAW, PW

            atoms = ase.io.read('{candidate_traj}')
            atoms.calc = GPAW(
                mode=PW(400),
                xc='PBE',
                kpts={{'size': (4, 4, 1), 'gamma': True}},
                occupations={{'name': 'fermi-dirac', 'width': 0.1}},
                convergence={{'energy': 1e-5}},
                txt='adslab.txt',
            )
            e = atoms.get_potential_energy()
            with open('adslab_result.json', 'w') as f:
                json.dump({{'energy': e}}, f)
        """)
        log.info("Running adslab single-point DFT (candidate rank %d)...", best_rank)
        e_adslab = run_gpaw_script(adslab_script, "adslab", run_dir, ncores)["energy"]
    log.info("E(adslab) = %.4f eV", e_adslab)

    # ── 3. Compute E_ads ──────────────────────────────────────────────────────
    # E_ads = E(adslab) - E(slab) - ½E(H₂)
    e_ads = e_adslab - e_slab - E_H2_HALF_REF
    log.info("E_ads (DFT/PBE) = %.4f eV", e_ads)
    log.info("E_ads (ML)      = %.4f eV", best_row["E_ads_ml_eV"])

    # ── 4. Save results ───────────────────────────────────────────────────────
    results = {
        "run_dir":           run_dir,
        "best_candidate":    best_rank,
        "h_position":        {"x": best_pos["x"], "y": best_pos["y"], "z": best_pos["z"]},
        "E_h2_half_ref_eV":  E_H2_HALF_REF,
        "E_slab_eV":         e_slab,
        "E_adslab_eV":       e_adslab,
        "E_ads_dft_eV":      e_ads,
        "E_ads_ml_eV":       best_row["E_ads_ml_eV"],
        "gibbs_free_ML_eV":  best_row["gibbs_free_ML"],
    }

    out_path = os.path.join(run_dir, "dft_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to an adsorbml_batch run folder")
    parser.add_argument("--ncores", type=int, default=DEFAULT_NCORES,
                        help=f"MPI ranks for GPAW (default: {DEFAULT_NCORES})")
    args = parser.parse_args()

    main(args.run_dir, args.ncores)
