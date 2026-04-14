"""
Cluster decoration for FairChem Slab objects.

Places a small homoatomic cluster of a given element on top of the surface,
producing a new Slab object ready for AdsorbML.

Unlike dopant substitution, cluster atoms sit atop the surface and retain
metal-metal bonding character; they do not integrate into the host lattice.
Surface atoms are identified by their ASE tag (tag == 1), which is set by
FairChem's tile_and_tag_atoms() during slab generation. The cluster is
centred above the xy centroid of the surface — a deterministic, geometry-
neutral placement that avoids edge/corner sites.

Cluster geometries (all planar, centred at the surface centroid):
  1 atom : single atom above the centroid
  2 atoms: dimer along the x-axis
  3 atoms: equilateral triangle in the xy plane
  4 atoms: square arrangement in the xy plane

Intra-cluster bond length is estimated as twice the covalent radius of the
cluster element (ASE Alvarez 2008 data). This slightly overestimates the
equilibrium metallic bond distance for 3d/4d transition metals (by ~5–15%;
e.g. 2×r_cov(Mo) = 3.08 Å vs bulk Mo NN = 2.73 Å). The UMA-M relaxation
corrects the geometry; the covalent-radius estimate only affects convergence
speed. Cluster atoms are tagged as surface atoms (tag == 1) so that AdsorbML
adsorbate-placement routines treat them as accessible adsorption sites.
"""

from __future__ import annotations

import numpy as np
from ase import Atom
from ase.data import atomic_numbers, covalent_radii
from fairchem.data.oc.core import Slab


def add_cluster(
    slab: Slab,
    element: str,
    count: int = 1,
    height: float = 2.5,
) -> Slab:
    """
    Return a new Slab with a `count`-atom cluster of `element` above the surface.

    Parameters
    ----------
    slab:
        Clean Slab object produced by Slab.from_bulk_get_specific_millers().
    element:
        Element symbol for the cluster atoms, e.g. "Pt".
    count:
        Number of cluster atoms. Supported values: 1, 2, 3, 4.
    height:
        Vertical clearance (Å) from the highest surface-tagged atom to the
        cluster plane. Defaults to 2.5 Å. For chalcogenide surfaces (MoS₂,
        MoSe₂) the topmost surface-tagged atoms are S/Se, which protrude
        ~1.6 Å above the Mo layer; a 2.5 Å clearance places the cluster
        ~4.1 Å above Mo, within the chemisorptive range once relaxed. For
        non-chalcogenide surfaces (MoP, Mo₂C, MoB, Mo₂N) the surface atoms
        are metals and 2.5 Å is similarly appropriate. The geometry will be
        refined during the subsequent UMA-M relaxation.

    Returns
    -------
    Slab
        New Slab object with cluster atoms appended and tagged as surface atoms
        (tag == 1). bulk is set to None because the composition no longer
        matches the parent bulk.

    Raises
    ------
    ValueError
        If the slab has no surface-tagged atoms, or if `count` is not in
        {1, 2, 3, 4}.
    """
    if count not in {1, 2, 3, 4}:
        raise ValueError(
            f"count={count} is not supported. Supported values: 1, 2, 3, 4."
        )

    atoms = slab.atoms.copy()
    tags = atoms.get_tags()
    surface_indices = [i for i, t in enumerate(tags) if t == 1]

    if not surface_indices:
        raise ValueError("Slab has no surface-tagged atoms (tag == 1).")

    # Centroid of surface atoms in xy — deterministic, geometry-neutral anchor.
    surface_xy = atoms.positions[surface_indices, :2]
    centroid_xy = surface_xy.mean(axis=0)

    # Place cluster plane above the highest surface atom.
    z_max = atoms.positions[surface_indices, 2].max()
    z_cluster = z_max + height

    # Intra-cluster bond length: 2 × covalent radius (Alvarez 2008 values).
    # This overestimates the metallic NN distance by ~5–15% for 3d/4d metals
    # (e.g. Mo: 3.08 Å vs bulk NN 2.73 Å). UMA-M relaxation corrects this.
    d = 2.0 * covalent_radii[atomic_numbers[element]]

    # Build xy offsets for each atom in the cluster, centred at origin.
    if count == 1:
        xy_offsets = np.array([[0.0, 0.0]])
    elif count == 2:
        # Dimer along x-axis.
        xy_offsets = np.array([[-d / 2, 0.0], [d / 2, 0.0]])
    elif count == 3:
        # Equilateral triangle; circumradius R = d / sqrt(3).
        R = d / np.sqrt(3)
        angles = np.radians([90.0, 210.0, 330.0])
        xy_offsets = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
    else:  # count == 4
        # Square with side d.
        h = d / 2
        xy_offsets = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])

    # Append cluster atoms, tagged as surface atoms so AdsorbML uses them.
    for dx, dy in xy_offsets:
        x, y = centroid_xy + np.array([dx, dy])
        atoms.append(Atom(element, position=(x, y, z_cluster), tag=1))

    return Slab(
        bulk=None,
        slab_atoms=atoms,
        millers=slab.millers,
        shift=slab.shift,
        top=slab.top,
        oriented_bulk=slab.oriented_bulk,
    )