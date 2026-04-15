# her-catalyst-screen

ML-accelerated adsorption energy screening for HER catalyst discovery, combining FairChem AdsorbML relaxations with GPAW DFT validation.

## Overview

This pipeline screens transition metal compounds for hydrogen evolution reaction (HER) activity by computing hydrogen adsorption energies (ΔG\*H) across a large combinatorial space of surfaces, terminations, and defect configurations. A sequential ML→DFT funnel concentrates expensive quantum-mechanical compute on the most promising candidates only.

**Defect types screened:** pristine surfaces, point vacancies, heteroatom dopants, supported metal clusters

## Pipeline

```
Materials Project / FairChem OC bulk DB
              ↓
   1-lookup_bulk_ids.py        — identify bulk polymorphs
              ↓
   2-create_materials.py       — generate slabs, apply defects, ML pre-relax (UMA-M/OMAT)
              ↓
   3-adsorb_ml.py              — AdsorbML *H screening, 100 candidates/slab (UMA-M/OC20)
              ↓
   4-gpaw_runner.py            — DFT single-point validation on best candidate (GPAW/PBE)
```

Each stage produces structured outputs consumed by the next, forming an auditable funnel from ~2400 slab variants down to DFT-validated adsorption energies.

## Requirements

- Python 3.10+
- [`fairchem`](https://github.com/FAIR-Chem/fairchem) — UMA-M model (`uma-m-1p1`), `FAIRChemCalculator`, `run_adsorbml`
- [`ase`](https://wiki.fysik.dtu.dk/ase/) — structure I/O, LBFGS optimizer, space group analysis
- [`polars`](https://pola.rs/) — DataFrame I/O
- `pyyaml` — config parsing
- CUDA-capable GPU (scripts 2 & 3)
- [`gpaw`](https://gpaw.readthedocs.io/) + MPI — DFT validation (script 4, optional)
- [`mp-api`](https://github.com/materialsproject/api) — Materials Project fallback lookup (script 1, optional)

Install into a dedicated environment:

```bash
python -m venv adsorbml_venv
source adsorbml_venv/bin/activate
pip install -r requirements.txt
# GPAW installation is system-specific; see https://gpaw.readthedocs.io/install.html
```

## Usage

Run scripts sequentially from the project root with the virtual environment active.

### Step 1 — Bulk lookup

Identifies bulk crystal structures for all target compounds from the FairChem OC bulk database, falling back to the Materials Project API for missing formulas.

```bash
export MP_API_KEY=<your_key>   # required only for Materials Project fallback
python 1-lookup_bulk_ids.py
```

Output: `lookups/bulk_ids.csv`

### Step 2 — Slab generation and ML pre-relaxation

Generates all symmetry-distinct slab terminations from each bulk, applies defect modifications (vacancies, dopants, clusters), and pre-relaxes every variant with UMA-M/OMAT via LBFGS (fmax = 0.02 eV/Å, max 200 steps).

```bash
python 2-create_materials.py
```

Outputs: `bulks/*.traj` (one file per slab variant), `adsorbml_prepared.csv` (manifest)

### Step 3 — AdsorbML hydrogen screening

For each pre-relaxed slab, generates 100 candidate \*H placements and relaxes them with UMA-M/OC20. Ranks candidates by proximity to thermoneutral adsorption (ΔG\*H = E_ads + 0.24 eV ≈ 0).

```bash
python 3-adsorb_ml.py
```

Outputs: `adsorbml_results/uma-m_run-1/{slab_label}/candidates.csv` + trajectory files, `adsorbml_results/uma-m_run-1/adsorbml_batch_summary.csv`

### Step 4 — GPAW DFT validation (optional)

Runs two GPAW single-point calculations (clean slab + adslab) on the best ML candidate for a given run directory. Uses PBE/PW400/4×4×1 k-points.

```bash
python 4-gpaw_runner.py adsorbml_results/uma-m_run-1/<slab_label> --ncores 8
```

Output: `<run_dir>/dft_results.json` with ML and DFT adsorption energies side by side.

## Configuration

Edit `screening_config.yaml` to control which compounds are screened and what defects are applied. A fully annotated schema reference is provided in `screening_config_template.yaml`.

| Field | Required | Description |
|---|---|---|
| `name` | yes | Chemical formula — must match a bulk in `lookups/bulk_ids.csv` |
| `vacancies` | no | List of `{element, count}` — surface atoms to remove (1 = monovacancy, 2 = paired) |
| `clusters` | no | List of `{element, count}` — homoatomic clusters above surface (1–4 atoms) |
| `dopants` | no | List of `{element[, target]}` — surface substitutions; `target` defaults to most abundant surface element |

## Key outputs

| File | Description |
|---|---|
| `lookups/bulk_ids.csv` | Bulk IDs, formulas, space groups |
| `adsorbml_prepared.csv` | Slab manifest: miller indices, termination, defect metadata, file path |
| `adsorbml_results/uma-m_run-1/adsorbml_batch_summary.csv` | All 100 candidates per slab with ML adsorption energies |
| `adsorbml_results/uma-m_run-1/<label>/candidates.csv` | Per-slab candidate ranking |
| `<run_dir>/dft_results.json` | DFT-validated adsorption energy for the best candidate |

## Methods summary

**ML pre-relaxation (step 2):** UMA-M v1.1 with the OMAT output head. Chosen over the OC20 head for defect-modified slabs because the OMAT training set broadly covers inorganic surfaces across the periodic table, including local bonding environments that diverge from the adsorbate-containing OC20 distribution.

**AdsorbML screening (step 3):** UMA-M v1.1 with the OC20 output head; energies referenced to gas-phase H2 via the same model for internal consistency. Candidate ranking uses the Sabatier criterion: ΔG\*H = E_ads + 0.24 eV, selecting the candidate closest to zero.

**DFT validation (step 4):** GPAW, PBE-GGA, PAW formalism, 400 eV plane-wave cutoff, 4×4×1 Monkhorst–Pack k-mesh, Fermi–Dirac smearing (0.1 eV). Single-point energies on ML-relaxed geometries. H2 reference: E(H2)/2 = −3.477 eV (from the AdsorbML paper).

## References

1. Jain et al., *APL Mater.* 1 (2013) 011002 — Materials Project
2. Larsen et al., *J. Phys.: Condens. Matter* 29 (2017) 273002 — ASE
3. Wood et al., *arXiv:2506.23971* (2025) — UMA model family
4. Chanussot et al., *ACS Catal.* 11 (2021) 6059 — OC20 / Open Catalyst Project
5. Faiyad & Martini, *arXiv:2510.05339* (2026) — UMA-M validation on doped MoS2
6. Lan et al., *npj Comput. Mater.* 9 (2023) 172 — AdsorbML
7. Enkovaara et al., *J. Phys.: Condens. Matter* 22 (2010) 253202 — GPAW
8. Mortensen et al., *J. Chem. Phys.* 160 (2024) 092503 — GPAW 24
9. Blöchl, *Phys. Rev. B* 50 (1994) 17953 — PAW method
10. Perdew, Burke & Ernzerhof, *Phys. Rev. Lett.* 77 (1996) 3865 — PBE functional
