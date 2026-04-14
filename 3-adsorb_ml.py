import os, ast
import polars as pl
import ase.io
from ase.optimize import LBFGS
from fairchem.data.oc.core import Slab
from fairchem.core.components.calculate.recipes.adsorbml import run_adsorbml
from fairchem.core import FAIRChemCalculator

calc = FAIRChemCalculator.from_model_checkpoint("uma-m-1p1", task_name="oc20", device="cuda")

ADSORBATE_SMILES = "*H"
SLABS_TO_VERIFY  = "adsorbml_prepared.csv"
OUT_DIR          = "adsorbml_results/uma-m_run-1"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load manifest ─────────────────────────────────────────────────────────
def _to_tuple(x):
    if isinstance(x, tuple):
        return x
    try:
        v = ast.literal_eval(str(x))
        return v if isinstance(v, tuple) else None
    except Exception:
        return None

df = pl.read_csv(SLABS_TO_VERIFY)
df = df.with_columns(pl.col("millers").map_elements(_to_tuple, return_dtype=pl.Object))

print(f"Found {len(df)} slab(s) to screen")

# ── Batch run ─────────────────────────────────────────────────────────────
all_results = []

for row in df.iter_rows(named=True):
    slab_file = row["slab_file"]
    slab_name = row["slab_name"]
    millers   = row["millers"]
    term_idx  = row["termination_idx"]

    # Run label derived from the pre-relaxed file written by 2-create_materials.py
    run_label = os.path.splitext(os.path.basename(slab_file))[0]
    run_dir   = os.path.join(OUT_DIR, run_label)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running: {run_label}")
    print(f"{'='*60}")

    # Load pre-relaxed slab (omat-relaxed by 2-create_materials.py)
    try:
        atoms = ase.io.read(slab_file)
        slab  = Slab(bulk=None, slab_atoms=atoms, millers=millers,
                     shift=None, top=None, oriented_bulk=None)
    except Exception as e:
        print(f"  [ERROR] Could not load slab: {e}")
        continue

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
        print(f"  [ERROR] run_adsorbml failed: {e}")
        continue

    candidates = outputs["adslabs"]

    if not candidates:
        print(f"  [SKIP] No adsorbate placements generated")
        continue

    # Save all candidate structures
    for i, cand in enumerate(candidates):
        ase.io.write(os.path.join(run_dir, f"candidate_{i}.traj"), cand["atoms"])

    # Build summary dataframe
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
    cand_df.write_csv(os.path.join(run_dir, "candidates.csv"))

    best_e = cand_df["E_ads_ml_eV"].min()
    print(f"  Best E_ads (ML) = {best_e:.4f} eV  ({len(candidates)} candidates)")
    all_results.append(cand_df)

# ── Consolidated summary ──────────────────────────────────────────────────
if all_results:
    summary_df = pl.concat(all_results)
    summary_path = os.path.join(OUT_DIR, "adsorbml_batch_summary.csv")
    summary_df.write_csv(summary_path)
    print(f"\nSaved consolidated results → {summary_path}")

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
    print("\nBest ML adsorption energies per system:")
    print(best)