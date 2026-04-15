"""
Dopant substitution for FairChem Slab objects.

Replaces one surface atom of a given element with a dopant element,
producing a new Slab object ready for AdsorbML.

Surface atoms are identified by their ASE tag (tag == 1), which is set
by FairChem's tile_and_tag_atoms() during slab generation. Among all
surface atoms of the target element, the one closest to the xy centroid
of the slab is substituted — a deterministic, geometry-neutral choice.
"""

from __future__ import annotations

import numpy as np
from fairchem.data.oc.core import Slab


def substitute_surface_atom(
    slab: Slab,
    dopant: str,
    target: str | None = None,
) -> Slab:
    """
    Return a new Slab with one surface atom substituted by `dopant`.

    Parameters
    ----------
    slab:
        Clean Slab object produced by Slab.from_bulk_get_specific_millers().
    dopant:
        Element symbol to insert, e.g. "Pt".
    target:
        Element symbol to replace. If None, defaults to the most abundant
        element among surface-tagged atoms (tag == 1).

    Returns
    -------
    Slab
        New Slab object with the substitution applied. bulk is set to None
        because the composition no longer matches the parent bulk.

    Raises
    ------
    ValueError
        If no surface atom of the target element is found.
    """
    atoms = slab.atoms.copy()
    tags = atoms.get_tags()
    surface_indices = [i for i, t in enumerate(tags) if t == 1]

    if not surface_indices:
        raise ValueError("Slab has no surface-tagged atoms (tag == 1).")

    if target is None:
        symbols = [atoms[i].symbol for i in surface_indices]
        target = max(set(symbols), key=symbols.count)

    candidates = [i for i in surface_indices if atoms[i].symbol == target]
    if not candidates:
        raise ValueError(
            f"No surface atom with element '{target}' found. "
            f"Surface elements present: {sorted({atoms[i].symbol for i in surface_indices})}"
        )

    # Pick the candidate closest to the xy centroid of all surface atoms.
    # This is deterministic and avoids edge/corner sites which can produce
    # spurious adsorption geometries.
    surface_xy = atoms.positions[surface_indices, :2]
    centroid_xy = surface_xy.mean(axis=0)
    candidate_xy = atoms.positions[candidates, :2]
    distances = np.linalg.norm(candidate_xy - centroid_xy, axis=1)
    replace_idx = candidates[int(np.argmin(distances))]

    atoms[replace_idx].symbol = dopant

    return Slab(
        bulk=None,
        slab_atoms=atoms,
        millers=slab.millers,
        shift=slab.shift,
        top=slab.top,
        oriented_bulk=slab.oriented_bulk,
    )
