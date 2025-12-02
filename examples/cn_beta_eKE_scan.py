from __future__ import annotations

"""Generate beta-anisotropy scans for CN over a kinetic-energy window."""

import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - script entrypoint
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from photodetachment_suite.DO_reader import DysonOrbitalBuilder, UniformGrid, load_qchem_output
from photodetachment_suite.averaging_angle_grids import get_angle_grid
from photodetachment_suite.beta_calculator import BetaResult, calculate_beta
from photodetachment_suite.grid import CartesianGrid
from photodetachment_suite.models import TransitionChannel


def build_cn_orbital_pair(output_path: Path, working_grid: UniformGrid):
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

    binding_energy_ev = 0.0
    e_ke = np.array([0.001, 0.01, 0.1, 0.5])
    photon_grid = binding_energy_ev + e_ke

    axis = np.linspace(-19.0, 19.0, 201)
    working_grid = UniformGrid(axis, axis, axis)

    orbital_pair, dyson = build_cn_orbital_pair(output_path, working_grid)
    cart_grid = CartesianGrid(dyson.grid.x, dyson.grid.y, dyson.grid.z)
    angle_grid = get_angle_grid("hard-coded")

    channel = TransitionChannel(
        label="CN 1/A1",
        binding_energy_ev=binding_energy_ev,
        orbitals=[orbital_pair],
        franck_condon=1.0,
    )

    plane_wave_result = calculate_beta(
        channels=[channel],
        grid=cart_grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid,
        continuum="plane_wave",
        integration_method="trapezoidal",
    )

    point_dipole_result = calculate_beta(
        channels=[channel],
        grid=cart_grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid,
        continuum="point_dipole",
        continuum_options={"dipole_strength": 0.3, "l_max": 5},
        integration_method="trapezoidal",
    )

    out_dir = suite_dir / "results" / "cn_beta"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_beta_csv(out_dir / "cn_beta_plane_wave.csv", plane_wave_result, binding_energy_ev)
    save_beta_csv(out_dir / "cn_beta_point_dipole.csv", point_dipole_result, binding_energy_ev)


if __name__ == "__main__":
    main()
