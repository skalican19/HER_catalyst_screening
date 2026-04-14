"""
Search the FairChem OC bulk database for compounds listed in screening_config.yaml.
For compounds not found locally, fetch candidates from Materials Project.

Usage:
    MP_API_KEY=<your_key> python lookup_bulk_ids.py

MP API key: https://next-gen.materialsproject.org/api
"""

import os
import polars as pl
import pickle
import yaml
from fairchem.data.oc.databases.pkls import BULK_PKL_PATH

with open("screening_config.yaml") as f:
    config = yaml.safe_load(f)

target_formulas = [c["name"] for c in config["compounds"]]

with open(BULK_PKL_PATH, "rb") as f:
    bulk_db = pickle.load(f)

hits: dict[str, list[tuple]] = {f: [] for f in target_formulas}

from ase.spacegroup import get_spacegroup

for entry in bulk_db:
    atoms = entry["atoms"]
    formula = atoms.get_chemical_formula(empirical=True)
    if formula in hits:
        try:
            sg = get_spacegroup(atoms, symprec=0.1)
            sg_no, sg_symbol = sg.no, sg.symbol
        except Exception:
            sg_no, sg_symbol = None, None
        hits[formula].append((entry["src_id"], sg_no, sg_symbol))

print(f"OC bulk database: {len(bulk_db)} entries\n")

csv_rows: list[dict] = []

missing = []
for formula in target_formulas:
    matches = hits[formula]
    if not matches:
        print(f"  {formula:<10}  NOT FOUND in OC database")
        missing.append(formula)
    else:
        print(f"  {formula:<10}  {len(matches)} match(es):")
        for src_id, sg_no, sg_symbol in matches:
            sg_str = f"sg={sg_no} ({sg_symbol})" if sg_no is not None else "sg=?"
            print(f"    {src_id}  {sg_str}")
            csv_rows.append({"formula": formula, "bulk-id": src_id, "sg-number": sg_no, "sg-symbol": sg_symbol})

if missing:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print("\nSet MP_API_KEY to fetch missing compounds from Materials Project.")
    else:
        print("\nFetching missing compounds from Materials Project...\n")
        try:
            from mp_api.client import MPRester
        except ImportError:
            print("mp-api not installed. Run: pip install mp-api")
            raise SystemExit(1)

        with MPRester(api_key) as mpr:
            for formula in missing:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "formula_pretty", "symmetry"],
                )
                if not docs:
                    print(f"  {formula:<10}  not found on Materials Project either")
                    continue
                print(f"  {formula:<10}  {len(docs)} MP candidate(s):")
                for doc in docs:
                    sg = doc.symmetry
                    print(f"    {doc.material_id}  {doc.formula_pretty}  sg={sg.number} ({sg.symbol})")
                    csv_rows.append({"formula": formula, "bulk-id": doc.material_id, "sg-number": sg.number, "sg-symbol": sg.symbol})

out_path = "./lookups/bulk_ids.csv"
pl.DataFrame(csv_rows, schema={"formula": pl.String, "bulk-id": pl.String, "sg-number": pl.Int32, "sg-symbol": pl.String}).write_csv(out_path)
print(f"\nSaved {len(csv_rows)} entries to {out_path}")
