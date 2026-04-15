"""
2-create_materials.py

For every bulk in lookups/bulk_ids.csv, enumerate all symmetry-distinct slab
surfaces and apply the defects (vacancies, clusters, dopants) defined in
screening_config.yaml.

Outputs
-------
bulks/
    One .traj file per (bulk_id, millers, termination_idx, defect_variant).
    Filename pattern:
        {formula}_{bulk_id}_{millerstr}_t{term}_clean.traj
        {formula}_{bulk_id}_{millerstr}_t{term}_vac{element}{count}.traj
        {formula}_{bulk_id}_{millerstr}_t{term}_cl{element}{count}.traj
        {formula}_{bulk_id}_{millerstr}_t{term}_dop{element}.traj

adsorbml_prepared.csv
    Manifest consumed by 3-adsorb_ml.py.  One row per slab file, with columns:
        slab_name, slab_id, sg_number, sg_symbol,
        millers, termination_idx,
        dopant, vacancy_element, vacancy_count,
        cluster_element, cluster_count,
        is_edge, is_interface, is_sheet,
        slab_file
"""

import os
import sys
import yaml
import ase.io
import polars as pl
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator
from fairchem.core.components.calculate.recipes.adsorbml import relax_job
from fairchem.data.oc.core import Bulk, Slab

sys.path.insert(0, os.path.dirname(__file__))
from utils.vacancy_slab import create_vacancy
from utils.cluster_slab import add_cluster
from utils.dope_slab import substitute_surface_atom

CONFIG_FILE   = "screening_config.yaml"
BULK_IDS_FILE = "lookups/bulk_ids.csv"
BULKS_DIR     = "bulks"
OUT_CSV       = "adsorbml_prepared.csv"

os.makedirs(BULKS_DIR, exist_ok=True)

omat_calc = FAIRChemCalculator.from_model_checkpoint(
    "uma-m-1p1", task_name="omat", device="cuda"
)

# ── Config & lookup ────────────────────────────────────────────────────────
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

compound_cfg = {c["name"]: c for c in config["compounds"]}

bulk_df = pl.read_csv(BULK_IDS_FILE)

# ── Main loop ──────────────────────────────────────────────────────────────
rows: list[dict] = []

for bulk_row in bulk_df.iter_rows(named=True):
    formula   = bulk_row["formula"]
    bulk_id   = bulk_row["bulk-id"]
    sg_number = bulk_row["sg-number"]
    sg_symbol = bulk_row["sg-symbol"]

    cc       = compound_cfg.get(formula, {})
    vacancies = cc.get("vacancies", [])
    clusters  = cc.get("clusters",  [])
    dopants   = cc.get("dopants",   [])

    print(f"\n{'='*60}")
    print(f"{formula}  {bulk_id}  ({sg_symbol})")
    print(f"{'='*60}")

    try:
        bulk      = Bulk(bulk_src_id_from_db=bulk_id)
        all_slabs = Slab.from_bulk_get_all_slabs(bulk)
    except Exception as exc:
        print(f"  [ERROR] slab generation failed: {exc}")
        continue

    print(f"  {len(all_slabs)} slab(s) total")

    # Group by miller indices so termination_idx is contiguous per orientation.
    millers_groups: dict[tuple, list] = {}
    for slab in all_slabs:
        millers_groups.setdefault(tuple(slab.millers), []).append(slab)

    for millers_tuple, term_slabs in millers_groups.items():
        miller_str = "".join(str(m) for m in millers_tuple)

        for term_idx, slab in enumerate(term_slabs):
            base = f"{formula}_{bulk_id}_{miller_str}_t{term_idx}"

            def _record(
                *,
                dopant=None,
                vacancy_element=None, vacancy_count=None,
                cluster_element=None, cluster_count=None,
                slab_file,
            ) -> dict:
                return dict(
                    slab_name=formula,
                    slab_id=bulk_id,
                    sg_number=sg_number,
                    sg_symbol=sg_symbol,
                    millers=str(millers_tuple),
                    termination_idx=term_idx,
                    dopant=dopant,
                    vacancy_element=vacancy_element,
                    vacancy_count=vacancy_count,
                    cluster_element=cluster_element,
                    cluster_count=cluster_count,
                    is_edge=False,
                    is_interface=False,
                    is_sheet=False,
                    slab_file=slab_file,
                )

            # -- Clean -------------------------------------------------------
            clean_path = os.path.join(BULKS_DIR, f"{base}_clean.traj")
            relaxed = relax_job(slab.atoms, calc=omat_calc, optimizer_cls=LBFGS, fmax=0.02, steps=200)
            ase.io.write(clean_path, relaxed["atoms"])
            rows.append(_record(slab_file=clean_path))
            print(f"  {base}_clean")

            # -- Vacancies ---------------------------------------------------
            for vac in vacancies:
                el, cnt = vac["element"], vac["count"]
                try:
                    vac_slab = create_vacancy(slab, target=el, count=cnt)
                    path = os.path.join(BULKS_DIR, f"{base}_vac{el}{cnt}.traj")
                    relaxed = relax_job(vac_slab.atoms, calc=omat_calc, optimizer_cls=LBFGS, fmax=0.02, steps=200)
                    ase.io.write(path, relaxed["atoms"])
                    rows.append(_record(vacancy_element=el, vacancy_count=cnt, slab_file=path))
                    print(f"  {base}_vac{el}{cnt}")
                except ValueError as exc:
                    print(f"  [SKIP] vac {el}×{cnt} on {base}: {exc}")

            # -- Clusters ----------------------------------------------------
            for cl in clusters:
                el, cnt = cl["element"], cl["count"]
                try:
                    cl_slab = add_cluster(slab, element=el, count=cnt)
                    path = os.path.join(BULKS_DIR, f"{base}_cl{el}{cnt}.traj")
                    relaxed = relax_job(cl_slab.atoms, calc=omat_calc, optimizer_cls=LBFGS, fmax=0.02, steps=200)
                    ase.io.write(path, relaxed["atoms"])
                    rows.append(_record(cluster_element=el, cluster_count=cnt, slab_file=path))
                    print(f"  {base}_cl{el}{cnt}")
                except (ValueError, KeyError) as exc:
                    print(f"  [SKIP] cluster {el}×{cnt} on {base}: {exc}")

            # -- Dopants -----------------------------------------------------
            for dop in dopants:
                el     = dop["element"]
                target = dop.get("target")
                try:
                    dop_slab = substitute_surface_atom(slab, dopant=el, target=target)
                    path = os.path.join(BULKS_DIR, f"{base}_dop{el}.traj")
                    relaxed = relax_job(dop_slab.atoms, calc=omat_calc, optimizer_cls=LBFGS, fmax=0.02, steps=200)
                    ase.io.write(path, relaxed["atoms"])
                    rows.append(_record(dopant=el, slab_file=path))
                    print(f"  {base}_dop{el}")
                except ValueError as exc:
                    print(f"  [SKIP] dopant {el} on {base}: {exc}")

# ── Write manifest ─────────────────────────────────────────────────────────
df = pl.DataFrame(
    rows,
    schema={
        "slab_name":        pl.String,
        "slab_id":          pl.String,
        "sg_number":        pl.Int32,
        "sg_symbol":        pl.String,
        "millers":          pl.String,
        "termination_idx":  pl.Int32,
        "dopant":           pl.String,
        "vacancy_element":  pl.String,
        "vacancy_count":    pl.Int32,
        "cluster_element":  pl.String,
        "cluster_count":    pl.Int32,
        "is_edge":          pl.Boolean,
        "is_interface":     pl.Boolean,
        "is_sheet":         pl.Boolean,
        "slab_file":        pl.String,
    },
)
df.write_csv(OUT_CSV)
print(f"\nWrote {len(df)} rows → {OUT_CSV}")
print(df.head(10))
