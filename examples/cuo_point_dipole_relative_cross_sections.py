from __future__ import annotations

"""Compute relative cross sections for CuO using the point-dipole continuum."""

from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":  # pragma: no cover - script entrypoint
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from photodetachment_suite.DO_reader import DysonOrbitalBuilder, UniformGrid, load_qchem_output
from photodetachment_suite.averaging_angle_grids import get_angle_grid
from photodetachment_suite.cross_sections import (
    RelativeCrossSectionResult,
    calculate_relative_cross_sections,
)
from photodetachment_suite.grid import CartesianGrid
from photodetachment_suite.models import TransitionChannel


def main() -> None:
    suite_dir = Path(__file__).resolve().parents[1]
    output_path = suite_dir / "CuO.out"

    vib_transitions = np.array(
        [
            [1.7780, 8.492010e-01],
            [1.8574, 5.006826e-01],
            [1.9367, 1.656282e-01],
        ]
    )
    binding_energies = vib_transitions[:, 0]
    franck_condon = vib_transitions[:, 1]

    photon_grid_ev = np.linspace(1.8, 2.3, 5)

    axis = np.linspace(-9.0, 9.0, 51)
    working_grid = UniformGrid(axis, axis, axis)

    data = load_qchem_output(output_path)
    builder = DysonOrbitalBuilder(data)
    left = builder.build_orbital("left_alpha_dyson_orbital__1_b1", working_grid, recenter=False)
    right = builder.build_orbital("right_alpha_dyson_orbital__1_b1", working_grid, recenter=False)
    pair = left.to_orbital_pair(right=right)

    cart_grid = CartesianGrid(left.grid.x, left.grid.y, left.grid.z)
    angle_grid = get_angle_grid("repulsion", n_orientations=80)

    channels: list[TransitionChannel] = []
    for idx, (binding, fc) in enumerate(zip(binding_energies, franck_condon, strict=True)):
        channel = TransitionChannel(
            label=f"v{idx}",
            binding_energy_ev=float(binding),
            orbitals=[pair],
            franck_condon=float(fc),
        )
        channels.append(channel)

    result: RelativeCrossSectionResult = calculate_relative_cross_sections(
        channels=channels,
        grid=cart_grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_grid_ev,
        continuum="point_dipole",
        continuum_options={"dipole_strength": 0.3, "l_max": 0},
        integration_method="trapezoidal",
    )

    out_dir = Path(__file__).parent
    csv_path = out_dir / "cuo_point_dipole_relative_cross_sections.csv"
    header = ["photon_energy_ev", *result.channel_labels]
    data_matrix = np.column_stack((result.photon_energies_ev, result.relative_cross_section))
    np.savetxt(csv_path, data_matrix, delimiter=",", header=",".join(header), comments="")
    print(f"Saved data to {csv_path}")


if __name__ == "__main__":
    main()
