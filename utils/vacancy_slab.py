"""
Vacancy creation for FairChem Slab objects.

Removes one or more surface atoms of a given element, producing a new Slab
object ready for AdsorbML.

Surface atoms are identified by their ASE tag (tag == 1), which is set by
FairChem's tile_and_tag_atoms() during slab generation. Among all surface
atoms of the target element, the one(s) closest to the xy centroid of the
slab are removed — a deterministic, geometry-neutral choice that avoids
edge/corner sites.
"""

from __future__ import annotations

import numpy as np
from fairchem.data.oc.core import Slab


def create_vacancy(
    slab: Slab,
    target: str,
    count: int = 1,
) -> Slab:
    """
    Return a new Slab with `count` surface atoms of `target` removed.

    Parameters
    ----------
    slab:
        Clean Slab object produced by Slab.from_bulk_get_specific_millers().
    target:
        Element symbol to remove, e.g. "C" or "Mo".
    count:
        Number of atoms to remove. Typically 1 (monovacancy) or 2 (divacancy).

    Returns
    -------
    Slab
        New Slab object with the vacancy applied. bulk is set to None because
        the composition no longer matches the parent bulk.

    Raises
    ------
    ValueError
        If the target element is not found on the surface, or if `count`
        exceeds the number of available surface atoms of that element.
    """
    atoms = slab.atoms.copy()
    tags = atoms.get_tags()
    surface_indices = [i for i, t in enumerate(tags) if t == 1]

    if not surface_indices:
        raise ValueError("Slab has no surface-tagged atoms (tag == 1).")

    candidates = [i for i in surface_indices if atoms[i].symbol == target]
    if not candidates:
        available = sorted({atoms[i].symbol for i in surface_indices})
        raise ValueError(
            f"No surface atom with element '{target}' found. "
            f"Surface elements present: {available}"
        )
    if count > len(candidates):
        raise ValueError(
            f"Requested {count} vacancies of '{target}' but only "
            f"{len(candidates)} surface atoms of that element exist."
        )

    # Sort candidates by distance to xy centroid; remove the `count` closest.
    # This is deterministic and avoids edge/corner sites which can produce
    # spurious adsorption geometries.
    surface_xy = atoms.positions[surface_indices, :2]
    centroid_xy = surface_xy.mean(axis=0)
    candidate_xy = atoms.positions[candidates, :2]
    distances = np.linalg.norm(candidate_xy - centroid_xy, axis=1)
    order = np.argsort(distances)
    remove_indices = {candidates[int(i)] for i in order[:count]}

    # Build a boolean keep-mask and apply it. ASE correctly filters
    # FixAtoms constraints when indexing an Atoms object this way.
    keep_mask = [i not in remove_indices for i in range(len(atoms))]
    atoms = atoms[keep_mask]

    return Slab(
        bulk=None,
        slab_atoms=atoms,
        millers=slab.millers,
        shift=slab.shift,
        top=slab.top,
        oriented_bulk=slab.oriented_bulk,
    )