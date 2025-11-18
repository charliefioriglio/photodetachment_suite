"""Compute anisotropy parameters (beta) for photodetachment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .averaging_angle_grids import AngleGrid
from .amplitudes import compute_orientational_amplitudes, weighted_intensity
from .continuum import ContinuumType
from .grid import CartesianGrid
from .integration import IntegrationMethod
from .models import TransitionChannel

_HARTREE_TO_EV = 27.2114
_SPEED_OF_LIGHT_AU = 137.035999084  # atomic units of velocity
_DEFAULT_POL = np.array([0.0, 0.0, 1.0])
_DEFAULT_K_PAR = np.array([0.0, 0.0, 1.0])
_DEFAULT_K_PERP = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
)


def _energy_iterator(photon_energies: np.ndarray, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(
            photon_energies,
            total=photon_energies.size,
            desc=desc,
            bar_format="{l_bar}{n_fmt}/{total_fmt}",
            leave=False,
        )
    except ImportError:  # pragma: no cover - optional dependency
        return photon_energies


@dataclass(frozen=True)
class BetaResult:
    energies_ev: np.ndarray
    beta: np.ndarray
    sigma_parallel: np.ndarray
    sigma_perpendicular: np.ndarray
    sigma_total: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "energies_ev": self.energies_ev,
            "beta": self.beta,
            "sigma_parallel": self.sigma_parallel,
            "sigma_perpendicular": self.sigma_perpendicular,
            "sigma_total": self.sigma_total,
        }


def calculate_beta(
    channels: Sequence[TransitionChannel],
    grid: CartesianGrid,
    angle_grid: AngleGrid,
    photon_energies_ev: Iterable[float],
    *,
    continuum: ContinuumType = "analytic",
    continuum_options: dict | None = None,
    integration_method: IntegrationMethod = "trapezoidal",
    polarization_lab: np.ndarray | None = None,
    k_parallel_lab: np.ndarray | None = None,
    k_perpendicular_lab: Sequence[np.ndarray] | None = None,
) -> BetaResult:
    """Compute beta over a sequence of kinetic energies.

    Parameters
    ----------
    channels:
        Photodetachment channels, each containing a binding energy, Franck–Condon
        factor, and one or more Dyson orbital pairs.
    grid:
        Spatial grid defining the evaluation volume.
    angle_grid:
        Euler angles and quadrature weights used for orientational averaging.
    photon_energies_ev:
        Iterable of photon energies in electron volts.
    continuum:
        Continuum model to evaluate (currently ``"analytic"`` or ``"plane_wave"``).
        continuum_options:
            Optional keyword arguments forwarded to the continuum factory (e.g.
            ``{"l_max": 4}`` for the plane-wave expansion model).
    integration_method:
        Numerical integration rule to apply (``"trapezoidal"`` or ``"simpson"``).
    polarization_lab:
        Lab-frame polarization vector (defaults to ``[0, 0, 1]``).
    k_parallel_lab, k_perpendicular_lab:
        Unit vectors defining the emission directions to probe. The parallel
        direction defaults to ``z`` while the perpendicular set defaults to ``x``
        and ``y``.
    """

    if not channels:
        raise ValueError("At least one transition channel must be supplied")

    pol = polarization_lab if polarization_lab is not None else _DEFAULT_POL
    k_par = k_parallel_lab if k_parallel_lab is not None else _DEFAULT_K_PAR
    k_perp = (
        tuple(k_perpendicular_lab)
        if k_perpendicular_lab is not None
        else _DEFAULT_K_PERP
    )

    pol = np.asarray(pol, dtype=float)
    k_par = np.asarray(k_par, dtype=float)
    k_perp = tuple(np.asarray(vec, dtype=float) for vec in k_perp)

    if pol.shape != (3,) or k_par.shape != (3,) or any(vec.shape != (3,) for vec in k_perp):
        raise ValueError("Polarization and k-vectors must be 3D unit vectors")

    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            raise ValueError("Zero-length direction vectors are not allowed")
        return vec / norm

    pol = _normalize(pol)
    k_par = _normalize(k_par)
    k_perp = tuple(_normalize(vec) for vec in k_perp)

    weights = angle_grid.weights
    photon_energies_ev = np.asarray(list(photon_energies_ev), dtype=float)

    sigma_par = np.zeros_like(photon_energies_ev)
    sigma_perp = np.zeros_like(photon_energies_ev)
    beta_vals = np.zeros_like(photon_energies_ev)
    sigma_total = np.zeros_like(photon_energies_ev)

    for idx, photon_ev in enumerate(_energy_iterator(photon_energies_ev, "beta")):
        photon_au = photon_ev / _HARTREE_TO_EV
        sigma_par_total = 0.0
        sigma_perp_total = 0.0

        for channel in channels:
            e_kinetic_ev = photon_ev - channel.binding_energy_ev
            if e_kinetic_ev <= 0.0:
                continue

            e_kinetic_au = e_kinetic_ev / _HARTREE_TO_EV
            k_mag = math.sqrt(2.0 * e_kinetic_au)
            prefactor = (
                channel.franck_condon_squared
                * 8.0
                * math.pi
                * k_mag
                * photon_au
                / _SPEED_OF_LIGHT_AU
            )

            k_par_scaled = k_par * k_mag
            k_perp_scaled = [vec * k_mag for vec in k_perp]

            channel_sigma_par = 0.0
            channel_sigma_perp = 0.0

            for orbital in channel.orbitals:
                amps_par = compute_orientational_amplitudes(
                    orbital,
                    grid,
                    angle_grid,
                    k_par_scaled,
                    pol,
                    continuum=continuum,
                    integration_method=integration_method,
                    continuum_options=continuum_options,
                )
                channel_sigma_par += weighted_intensity(amps_par, weights)

                amps_perp = [
                    compute_orientational_amplitudes(
                        orbital,
                        grid,
                        angle_grid,
                        vec,
                        pol,
                        continuum=continuum,
                        integration_method=integration_method,
                        continuum_options=continuum_options,
                    )
                    for vec in k_perp_scaled
                ]
                channel_sigma_perp += sum(
                    weighted_intensity(amps, weights) for amps in amps_perp
                ) / len(k_perp_scaled)

            sigma_par_total += prefactor * channel_sigma_par
            sigma_perp_total += prefactor * channel_sigma_perp

        denom = sigma_par_total + 2.0 * sigma_perp_total
        sigma_par[idx] = float(np.real_if_close(sigma_par_total))
        sigma_perp[idx] = float(np.real_if_close(sigma_perp_total))
        beta_vals[idx] = (
            2.0 * (sigma_par_total - sigma_perp_total) / denom
            if denom != 0.0
            else 0.0
        )
        sigma_total[idx] = float(np.real_if_close((4.0 * math.pi / 3.0) * denom))

    return BetaResult(
        energies_ev=photon_energies_ev,
        beta=beta_vals,
        sigma_parallel=sigma_par,
        sigma_perpendicular=sigma_perp,
        sigma_total=sigma_total,
    )


__all__ = ["BetaResult", "calculate_beta"]
