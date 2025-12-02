"""Comprehensive CN Dyson-orbital smoke test.

This script showcases the major public entry points in ``photodetachment_suite``
using the analytic CN Dyson orbital supplied in ``CN_do.py``. The underlying
Gaussian coefficients originate from the ezDyson reference implementation
included in ``ezSuite/ezDyson_2021/ezdyson_code``; this demo illustrates how the
rewritten toolkit wraps those legacy assets into a modern, testable workflow.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import numpy as np

if __package__ is None or __package__ == "":  # pragma: no cover - script entrypoint
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from photodetachment_suite import CN_do
from photodetachment_suite.averaging_angle_grids import AngleGrid, get_angle_grid
from photodetachment_suite.beta_calculator import calculate_beta
from photodetachment_suite.cross_sections import (
    CrossSectionResult,
    calculate_relative_cross_sections,
    calculate_total_cross_sections,
)
from photodetachment_suite.grid import CartesianGrid
from photodetachment_suite.models import OrbitalPair, TransitionChannel
from photodetachment_suite.physical_dipole.cross_sections import (
    calculate_physical_dipole_cross_sections,
)


def build_demo_grid(axis_extent_bohr: float = 10.0, n_points: int = 41) -> tuple[CartesianGrid, float]:
    """Return a Cartesian grid and volume element for volumetric integrals."""

    axis = np.linspace(-axis_extent_bohr, axis_extent_bohr, n_points)
    grid = CartesianGrid(axis, axis, axis)
    dx, dy, dz = grid.spacing
    return grid, dx * dy * dz


def construct_cn_orbital_pair(grid: CartesianGrid, dV: float) -> OrbitalPair:
    """Build the left/right Dyson orbitals defined by :mod:`CN_do`."""

    left = CN_do.build_DO(CN_do.DO_coeffs_L, grid.x, grid.y, grid.z, dV, recenter=True)
    right = CN_do.build_DO(CN_do.DO_coeffs_R, grid.x, grid.y, grid.z, dV, recenter=True)

    atom_centers = np.stack((CN_do.R_C, CN_do.R_N))
    midpoint = atom_centers.mean(axis=0)
    bond_axis = atom_centers[0] - atom_centers[1]
    half_bond_length = 0.5 * float(np.linalg.norm(atom_centers[0] - atom_centers[1]))

    metadata = {
        "atom_centers_bohr": atom_centers,
        "bond_axis": bond_axis,
        "bond_midpoint_bohr": midpoint,
        "half_bond_length_bohr": half_bond_length,
    }

    return OrbitalPair(
        left=left,
        right=right,
        left_norm=CN_do.norm_L,
        right_norm=CN_do.norm_R,
        metadata=metadata,
    )


def build_demo_channels(orbital: OrbitalPair) -> Sequence[TransitionChannel]:
    """Fabricate two vibrational channels that share the same Dyson orbital."""

    ground = TransitionChannel(
        label="CN_v0",
        binding_energy_ev=1.5,
        orbitals=[orbital],
        franck_condon=1.0,
    )
    excited = TransitionChannel(
        label="CN_v1",
        binding_energy_ev=1.8,
        orbitals=[orbital],
        franck_condon=0.6,
    )
    return (ground, excited)


def summarize_result(result: CrossSectionResult, label: str) -> None:
    """Pretty-print a concise table of cross sections."""

    print(f"\n{label}")
    print("Energy (eV)  Total (arb)  " + "  ".join(f"{name:>10}" for name in result.channel_labels))
    for idx, energy in enumerate(result.photon_energies_ev):
        channel_vals = "  ".join(f"{result.per_channel[idx, ch]:10.3e}" for ch in range(len(result.channel_labels)))
        print(f"{energy:11.3f}  {result.total_cross_section[idx]:12.3e}  {channel_vals}")


def main() -> None:
    """Execute the sample workflow end-to-end and print intermediate results."""

    grid, dV = build_demo_grid()
    angle_grid: AngleGrid = get_angle_grid("simple", n_orientations=12, seed=2024)
    orbital = construct_cn_orbital_pair(grid, dV)
    channels = build_demo_channels(orbital)
    photon_grid_ev = np.linspace(2.1, 2.6, 4)

    plane_wave = calculate_total_cross_sections(
        channels=channels,
        grid=grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid_ev,
        continuum="analytic",
        integration_method="trapezoidal",
    )
    summarize_result(plane_wave, "Plane-wave total cross sections")

    relative = calculate_relative_cross_sections(
        channels=channels,
        grid=grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid_ev,
        continuum="plane_wave",
        integration_method="trapezoidal",
    )
    print("\nRelative channel weights (plane-wave continuum):")
    print(relative.relative_cross_section)

    beta = calculate_beta(
        channels=[channels[0]],
        grid=grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid_ev,
        continuum="analytic",
        integration_method="trapezoidal",
    )
    print("\nBeta scan for CN_v0 (plane wave):")
    print(np.column_stack((beta.energies_ev, beta.beta)))

    metadata = orbital.metadata or {}
    half_bond_length = metadata.get("half_bond_length_bohr")
    if half_bond_length is None:
        raise RuntimeError("CN metadata missing half_bond_length_bohr")

    dipole_strength = 0.57 / (2.0 * half_bond_length)
    physical_options = {
        "l_max": 4,
        "n_xi": 40,
        "n_eta": 40,
        "radial_max_terms": 30,
        "sum_modes": True,
        "max_modes_per_m": 4,
        "max_m": 4,
    }

    physical = calculate_physical_dipole_cross_sections(
        channels=[channels[0]],
        grid=grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid_ev,
        dipole_strength=dipole_strength,
        continuum_options=physical_options,
        integration_method="trapezoidal",
    )
    summarize_result(physical, "Physical-dipole total cross sections (CN_v0)")


if __name__ == "__main__":  # pragma: no cover - manual exercise
    main()
