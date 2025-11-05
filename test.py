from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from photodetachment_suite.DO_reader.parser import load_qchem_output
from photodetachment_suite.DO_reader.dyson import DysonBuildResult, DysonOrbitalBuilder, UniformGrid
from photodetachment_suite.averaging_angle_grids import get_angle_grid
from photodetachment_suite.grid import CartesianGrid
from photodetachment_suite.models import OrbitalPair, TransitionChannel
from photodetachment_suite.beta_calculator import calculate_beta
from photodetachment_suite.plots import plot_beta  # optional


def build_orbital_pair(path: Path, working_grid: UniformGrid) -> tuple[OrbitalPair, DysonBuildResult]:
    print(f"\nParsing Dyson orbitals from {path.name}")
    data = load_qchem_output(path)
    builder = DysonOrbitalBuilder(data)
    available = builder.available_orbitals()
    print("  Available labels:", available)

    left_label = next((label for label in available if label.startswith("left")), available[0])
    right_label = next((label for label in available if label.startswith("right")), None)

    left = builder.build_orbital(left_label, working_grid)
    right = builder.build_orbital(right_label, working_grid) if right_label else None

    pair = left.to_orbital_pair(right=right)
    return pair, left


def ensure_common_grid(dyson_results: Sequence[DysonBuildResult]) -> CartesianGrid:
    base = dyson_results[0].grid
    for other in dyson_results[1:]:
        if not (
            np.allclose(other.grid.x, base.x)
            and np.allclose(other.grid.y, base.y)
            and np.allclose(other.grid.z, base.z)
        ):
            raise ValueError(
                "Dyson orbitals were not built on a common grid; disable recentering or reuse a shared grid."
            )
    return CartesianGrid(base.x, base.y, base.z)


def main(orbitals: Iterable[Path]) -> None:
    axis = np.linspace(-10.0, 10.0, 51)
    working_grid = UniformGrid(axis, axis, axis)

    orbital_pairs: list[OrbitalPair] = []
    dyson_details: list[DysonBuildResult] = []

    for path in orbitals:
        pair, detail = build_orbital_pair(path, working_grid)
        orbital_pairs.append(pair)
        dyson_details.append(detail)

    cart_grid = ensure_common_grid(dyson_details)

    channel = TransitionChannel(
        label="CN",
        binding_energy_ev=0.0,
        orbitals=tuple(orbital_pairs),
        franck_condon=1.0,
    )

    angle_grid = get_angle_grid("repulsion", n_orientations=50) # "hard-coded", "repulsion", or "simple"
    e_ke = np.array([0.0001, 0.001, 0.005, 0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    photon_energies = channel.binding_energy_ev + e_ke

    result = calculate_beta(
        channels=[channel],
        grid=cart_grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_energies,
        continuum="point_dipole",
        continuum_options={"dipole_strength": 0.57, "l_max": 3},
        integration_method="trapezoidal",
    )

    print("\nphoton (eV)  beta")
    for energy, beta in zip(result.energies_ev, result.beta):
        print(f"{energy:8.3f}  {beta:6.3f}")

    # Optional: quick plot
    plot_beta(result)
    import matplotlib.pyplot as plt; plt.show()


if __name__ == "__main__":
    suite_dir = Path(__file__).resolve().parent
    output_files = [suite_dir / "CN.out"]
    main(output_files)