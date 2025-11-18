from __future__ import annotations

"""Demonstrate beta calculation for CN using the summed physical-dipole continuum."""

from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":  # pragma: no cover - script entrypoint
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from photodetachment_suite.DO_reader import DysonOrbitalBuilder, UniformGrid, load_qchem_output
from photodetachment_suite.averaging_angle_grids import get_angle_grid
from photodetachment_suite.beta_calculator import calculate_beta
from photodetachment_suite.grid import CartesianGrid
from photodetachment_suite.models import TransitionChannel


def main() -> None:
    suite_dir = Path(__file__).resolve().parents[1]
    output_path = suite_dir / "CN.out"

    axis = np.linspace(-19.0, 19.0, 151)
    working_grid = UniformGrid(axis, axis, axis)

    data = load_qchem_output(output_path)
    builder = DysonOrbitalBuilder(data)
    left = builder.build_orbital("left_alpha_dyson_orbital__1_a1", working_grid, recenter=False)
    right = builder.build_orbital("right_alpha_dyson_orbital__1_a1", working_grid, recenter=False)
    pair = left.to_orbital_pair(right=right)

    cart_grid = CartesianGrid(left.grid.x, left.grid.y, left.grid.z)
    angle_grid = get_angle_grid("repulsion", n_orientations=25)

    metadata = dict(pair.metadata)
    half_bond_length = metadata.get("half_bond_length_bohr")
    if half_bond_length is None:
        raise ValueError("Dyson metadata missing half_bond_length_bohr for CN")

    # Example dipole strength; adjust as needed for the molecule under study.
    dipole_strength = 0.57 / (2.0 * half_bond_length)

    channel = TransitionChannel(
        label="CN",
        binding_energy_ev=0.0,
        orbitals=[pair],
        franck_condon=1.0,
    )

    photon_grid_ev = np.array([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])

    result = calculate_beta(
        channels=[channel],
        grid=cart_grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid_ev,
        continuum="physical_dipole",
        continuum_options={
            "dipole_strength": dipole_strength,
            "l_max": 5,
            "sum_modes": True,
            "max_modes_per_m": 5,
            "max_m": 5,
        },
        integration_method="trapezoidal",
    )

    out_dir = Path(__file__).parent
    csv_path = out_dir / "cn_beta_physical_dipole.csv"
    header = ["photon_energy_ev", "beta", "sigma_parallel", "sigma_perpendicular", "sigma_total"]
    data_matrix = np.column_stack(
        (
            result.energies_ev,
            result.beta,
            result.sigma_parallel,
            result.sigma_perpendicular,
            result.sigma_total,
        )
    )
    np.savetxt(csv_path, data_matrix, delimiter=",", header=",".join(header), comments="")
    print(f"Saved beta data to {csv_path}")


if __name__ == "__main__":
    main()
