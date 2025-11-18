from __future__ import annotations

"""Compute and plot relative cross sections for the CuO 1/B1 Dyson orbital.

This recreates the ``cross_sections_gallup`` workflow using the refactored
``photodetachment_suite`` modules.
"""

from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":  # pragma: no cover - script entrypoint
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from photodetachment_suite.DO_reader import DysonOrbitalBuilder, UniformGrid, load_qchem_output
from photodetachment_suite.physical_dipole.wavefunction import (
    PhysicalDipoleParameters,
    build_wavefunction_in_body_frame,
    prepare_body_frame,
)


HARTREE_TO_EV = 27.2114
SPEED_OF_LIGHT_AU = 137.035999084


def integrate(volume_element: float, field: np.ndarray) -> complex:
    return np.sum(field) * volume_element


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
    trans_energies = vib_transitions[:, 0]
    franck_condon = vib_transitions[:, 1]

    photon_grid_ev = np.linspace(1.8, 2.0, 5)
    L_max = 2

    axis = np.linspace(-10.0, 10.0, 100)
    working_grid = UniformGrid(axis, axis, axis)

    data = load_qchem_output(output_path)
    builder = DysonOrbitalBuilder(data)
    left = builder.build_orbital("left_alpha_dyson_orbital__1_b1", working_grid, recenter=False)
    right = builder.build_orbital("right_alpha_dyson_orbital__1_b1", working_grid, recenter=False)
    pair = left.to_orbital_pair(right=right)

    X, Y, Z = working_grid.mesh()
    dV = working_grid.differential_volume()
    left_wave = pair.scaled_left()
    right_wave = pair.scaled_right()

    metadata = dict(pair.metadata)
    half_bond_length = metadata.get("half_bond_length_bohr")
    bond_axis = metadata.get("bond_axis")
    midpoint = metadata.get("bond_midpoint_bohr")
    if half_bond_length is None or bond_axis is None or midpoint is None:
        raise ValueError("Dyson metadata missing geometry information required for physical dipole continuum")

    dipole_strength = 1.7 / (2.0 * half_bond_length)

    X_body, Y_body, Z_body = prepare_body_frame(
        X,
        Y,
        Z,
        midpoint=np.asarray(midpoint, dtype=float),
        bond_axis=np.asarray(bond_axis, dtype=float),
    )

    base_params = dict(
        dipole_strength=float(dipole_strength),
        half_bond_length_bohr=float(half_bond_length),
        l_max=L_max,
        xi_match=2.5,
        radial_max_terms=80,
        n_xi=220,
        n_eta=180,
        n_phi=120,
    )

    polarizations = [X, Y, Z]
    cache: Dict[Tuple[int, int, float], np.ndarray] = {}

    def continuum_wavefunction(m: int, n: int, energy_au: float) -> np.ndarray | None:
        if energy_au <= 0.0:
            return None
        key = (m, n, round(energy_au, 9))
        if key in cache:
            return cache[key]
        params = PhysicalDipoleParameters(m_quantum=m, n_mode=n, **base_params)
        try:
            psi = build_wavefunction_in_body_frame(params, energy_au, X_body, Y_body, Z_body)
        except ValueError:
            return None
        cache[key] = psi
        return psi

    n_channels = len(trans_energies)
    sigma = np.zeros((n_channels, photon_grid_ev.size), dtype=float)

    for energy_idx, photon_ev in enumerate(photon_grid_ev):
        photon_au = photon_ev / HARTREE_TO_EV
        for chan_idx, (binding_ev, fc) in enumerate(zip(trans_energies, franck_condon, strict=True)):
            eKE_ev = photon_ev - binding_ev
            if eKE_ev <= 0.0:
                continue
            energy_au = eKE_ev / HARTREE_TO_EV
            k_mag = float(np.sqrt(2.0 * energy_au))
            prefactor = (fc ** 2) * 8.0 * np.pi * k_mag * photon_au / SPEED_OF_LIGHT_AU

            total_A = 0.0 + 0.0j
            for m in range(-L_max, L_max + 1):
                n_max = L_max - abs(m)
                for n_mode in range(n_max + 1):
                    psi = continuum_wavefunction(m, n_mode, energy_au)
                    if psi is None:
                        continue
                    cklm = 0.0 + 0.0j
                    for mu in polarizations:
                        dipole = mu
                        integrand_L = np.conj(psi) * dipole * left_wave
                        integrand_R = np.conj(right_wave) * dipole * psi
                        amp_L = integrate(dV, integrand_L)
                        amp_R = integrate(dV, integrand_R)
                        cklm += amp_L * amp_R
                    total_A += cklm / len(polarizations)

            sigma_val = prefactor * total_A
            sigma[chan_idx, energy_idx] = float(np.real_if_close(sigma_val))

    total_sigma = sigma.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(total_sigma > 0.0, sigma / total_sigma, 0.0)

    fig, ax = plt.subplots()
    for idx in range(n_channels):
        label = f"v={idx} ({trans_energies[idx]:.3f} eV)"
        ax.plot(photon_grid_ev, relative[idx], label=label)

    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Relative cross section")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("CuO 1/B1 relative cross sections (physical dipole)")
    ax.legend(title="Channel")
    fig.tight_layout()

    out_dir = Path(__file__).parent
    figure_path = out_dir / "cuo_relative_cross_sections.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    csv_path = out_dir / "cuo_relative_cross_sections.csv"
    header = ["photon_energy_ev", "v0", "v1", "v2"]
    data = np.column_stack((photon_grid_ev, relative.T))
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")
    print(f"Saved figure to {figure_path}")
    print(f"Saved data to {csv_path}")


if __name__ == "__main__":
    main()
