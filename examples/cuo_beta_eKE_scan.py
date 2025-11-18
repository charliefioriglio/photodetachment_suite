from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

if __package__ in {None, ""}:  # pragma: no cover - script entrypoint
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from photodetachment_suite.DO_reader import DysonOrbitalBuilder, UniformGrid, load_qchem_output
from photodetachment_suite.averaging_angle_grids import get_angle_grid
from photodetachment_suite.beta_calculator import BetaResult, calculate_beta
from photodetachment_suite.grid import CartesianGrid
from photodetachment_suite.models import TransitionChannel


def build_CuO_orbital_pair(output_path: Path, working_grid: UniformGrid):
    data = load_qchem_output(output_path)
    builder = DysonOrbitalBuilder(data)
    left = builder.build_orbital(
        "left_alpha_dyson_orbital__1_a1",
        working_grid,
        recenter=False,
    )
    right = builder.build_orbital(
        "right_alpha_dyson_orbital__1_a1",
        working_grid,
        recenter=False,
    )
    return left.to_orbital_pair(right=right), left


def save_beta_csv(path: Path, result: BetaResult, binding_energy_ev: float) -> None:
    e_ke = result.energies_ev - binding_energy_ev
    payload = np.column_stack(
        (
            e_ke,
            result.energies_ev,
            result.beta,
            result.sigma_parallel,
            result.sigma_perpendicular,
            result.sigma_total,
        )
    )
    header = "eKE_ev,photon_energy_ev,beta,sigma_parallel,sigma_perpendicular,sigma_total"
    np.savetxt(path, payload, delimiter=",", header=header, comments="")
    print(f"Saved beta scan to {path}")

def main() -> None:
    suite_dir = Path(__file__).resolve().parents[1]
    output_path = suite_dir / "CN.out"

    binding_energy_ev = 0
    e_ke = np.array([0.001, 0.01, 0.1, 0.3])
    photon_grid = binding_energy_ev + e_ke

    axis = np.linspace(-19.0, 19.0, 100)
    working_grid = UniformGrid(axis, axis, axis)

    orbital_pair, dyson = build_CuO_orbital_pair(output_path, working_grid)
    cart_grid = CartesianGrid(dyson.grid.x, dyson.grid.y, dyson.grid.z)
    angle_grid = get_angle_grid("repulsion", n_orientations=50)

    channel = TransitionChannel(
        label="CuO 1/A1",
        binding_energy_ev=binding_energy_ev,
        orbitals=[orbital_pair],
        franck_condon=1.0,
    )

    point_dipole_0_result = calculate_beta(
        channels=[channel],
        grid=cart_grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid,
        continuum="point_dipole",
        continuum_options={"dipole_strength": 0.0, "l_max": 3},
        integration_method="trapezoidal",
    )

    point_dipole_1_result = calculate_beta(
        channels=[channel],
        grid=cart_grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid,
        continuum="point_dipole",
        continuum_options={"dipole_strength": 0.05, "l_max": 3},
        integration_method="trapezoidal",
    )

    point_dipole_3_result = calculate_beta(
        channels=[channel],
        grid=cart_grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid,
        continuum="point_dipole",
        continuum_options={"dipole_strength": 0.15, "l_max": 3},
        integration_method="trapezoidal",
    )

    point_dipole_5_result = calculate_beta(
        channels=[channel],
        grid=cart_grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid,
        continuum="point_dipole",
        continuum_options={"dipole_strength": 0.25, "l_max": 3},
        integration_method="trapezoidal",
    )

    out_dir = suite_dir / "results" / "cn_beta"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_beta_csv(out_dir / "cuo_beta_point_0_dipole.csv", point_dipole_0_result, binding_energy_ev)
    save_beta_csv(out_dir / "cuo_beta_point_1_dipole.csv", point_dipole_1_result, binding_energy_ev)
    save_beta_csv(out_dir / "cuo_beta_point_3_dipole.csv", point_dipole_3_result, binding_energy_ev)
    save_beta_csv(out_dir / "cuo_beta_point_5_dipole.csv", point_dipole_5_result, binding_energy_ev)


if __name__ == "__main__":
    main()
