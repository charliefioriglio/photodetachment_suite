"""Total and relative cross section calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .amplitudes import compute_orientational_amplitudes, weighted_intensity
from .averaging_angle_grids import AngleGrid
from .continuum import ContinuumType
from .grid import CartesianGrid
from .integration import IntegrationMethod
from .models import TransitionChannel

_HARTREE_TO_EV = 27.2114
_SPEED_OF_LIGHT_AU = 137.035999084
_DEFAULT_POL = np.array([0.0, 0.0, 1.0])
_DEFAULT_K_PAR = np.array([0.0, 0.0, 1.0])
_DEFAULT_K_PERP = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
)


def _energy_iterator(photon_energies: np.ndarray, desc: str):
    """Wrap a photon-energy iterable with tqdm when available."""

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
class CrossSectionResult:
    """Container for total and per-channel cross sections."""

    photon_energies_ev: np.ndarray
    total_cross_section: np.ndarray
    per_channel: np.ndarray
    channel_labels: tuple[str, ...]

    def relative(self) -> np.ndarray:
        """Return relative channel weights at each photon energy."""

        totals = self.total_cross_section[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(totals > 0.0, self.per_channel / totals, 0.0)
        return rel

    def as_dict(self) -> dict[str, np.ndarray]:
        data = {
            "photon_energies_ev": self.photon_energies_ev,
            "total_cross_section": self.total_cross_section,
        }
        for idx, label in enumerate(self.channel_labels):
            data[f"channel_{label}_cross_section"] = self.per_channel[:, idx]
        return data


def calculate_total_cross_sections(
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
) -> CrossSectionResult:
    """Return absolute cross sections for each photon energy.

    Parameters
    ----------
    channels:
        Photodetachment channels to include in the sum.
    continuum:
        Continuum model identifier. ``"expansion"`` requires ``continuum_options``
        to provide ``l_max``.
    continuum_options:
        Optional keyword arguments forwarded to the continuum factory.
    """

    if not channels:
        raise ValueError("At least one transition channel must be supplied")

    pol = np.asarray(polarization_lab if polarization_lab is not None else _DEFAULT_POL, dtype=float)
    k_par = np.asarray(k_parallel_lab if k_parallel_lab is not None else _DEFAULT_K_PAR, dtype=float)
    k_perp = tuple(
        np.asarray(vec, dtype=float)
        for vec in (k_perpendicular_lab if k_perpendicular_lab is not None else _DEFAULT_K_PERP)
    )

    if pol.shape != (3,) or k_par.shape != (3,) or any(vec.shape != (3,) for vec in k_perp):
        raise ValueError("Polarization and k-vectors must be three-dimensional")

    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            raise ValueError("Zero-length direction vectors are not allowed")
        return vec / norm

    pol = _normalize(pol)
    k_par = _normalize(k_par)
    k_perp = tuple(_normalize(vec) for vec in k_perp)
    if not k_perp:
        raise ValueError("At least one perpendicular direction vector is required")

    photon_energies_ev = np.asarray(list(photon_energies_ev), dtype=float)
    n_energy = photon_energies_ev.size
    n_channel = len(channels)

    per_channel = np.zeros((n_energy, n_channel), dtype=float)
    total = np.zeros(n_energy, dtype=float)
    angle_weights = angle_grid.weights

    for energy_idx, photon_ev in enumerate(_energy_iterator(photon_energies_ev, "total σ")):
        photon_au = photon_ev / _HARTREE_TO_EV
        for channel_idx, channel in enumerate(channels):
            eKE_ev = photon_ev - channel.binding_energy_ev
            if eKE_ev <= 0.0:
                continue

            eKE_au = eKE_ev / _HARTREE_TO_EV
            k_mag = math.sqrt(2.0 * eKE_au)
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
                channel_sigma_par += weighted_intensity(amps_par, angle_weights)

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
                    weighted_intensity(amps, angle_weights) for amps in amps_perp
                ) / len(k_perp_scaled)

            sigma_channel = prefactor * channel_sigma_par
            sigma_channel_perp = prefactor * channel_sigma_perp
            value = (
                (4.0 * math.pi / 3.0)
                * (sigma_channel + 2.0 * sigma_channel_perp)
            )
            per_channel[energy_idx, channel_idx] = float(np.real_if_close(value))

        total_value = per_channel[energy_idx].sum()
        total[energy_idx] = float(np.real_if_close(total_value))

    labels = tuple(channel.label for channel in channels)
    return CrossSectionResult(
        photon_energies_ev=photon_energies_ev,
        total_cross_section=total,
        per_channel=per_channel,
        channel_labels=labels,
    )
def calculate_relative_cross_sections_from_total(result: CrossSectionResult) -> np.ndarray:
    """Return channel fractions vs photon energy from a precomputed result."""

    return result.relative()


@dataclass(frozen=True)
class RelativeCrossSectionResult:
    """Relative (per-channel) cross sections as fractions of the total."""

    photon_energies_ev: np.ndarray
    relative_cross_section: np.ndarray
    channel_labels: tuple[str, ...]

    def as_dict(self) -> dict[str, np.ndarray]:
        data = {"photon_energies_ev": self.photon_energies_ev}
        for idx, label in enumerate(self.channel_labels):
            data[f"channel_{label}_relative_cross_section"] = self.relative_cross_section[:, idx]
        return data


def calculate_relative_cross_sections(
    channels: Sequence[TransitionChannel],
    grid: CartesianGrid,
    angle_grid: AngleGrid,
    photon_energies_ev: Iterable[float],
    *,
    continuum: ContinuumType,
    continuum_options: dict | None = None,
    integration_method: IntegrationMethod = "trapezoidal",
    polarization_lab: np.ndarray | None = None,
    k_parallel_lab: np.ndarray | None = None,
    k_perpendicular_lab: Sequence[np.ndarray] | None = None,
) -> RelativeCrossSectionResult:
    """Return relative cross sections per channel for non-analytic continua."""

    if continuum == "analytic":
        raise ValueError("Relative cross sections are not available for the analytic continuum model")

    total_result = calculate_total_cross_sections(
        channels=channels,
        grid=grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_energies_ev,
        continuum=continuum,
        continuum_options=continuum_options,
        integration_method=integration_method,
        polarization_lab=polarization_lab,
        k_parallel_lab=k_parallel_lab,
        k_perpendicular_lab=k_perpendicular_lab,
    )

    relative = calculate_relative_cross_sections_from_total(total_result)
    return RelativeCrossSectionResult(
        photon_energies_ev=total_result.photon_energies_ev,
        relative_cross_section=relative,
        channel_labels=total_result.channel_labels,
    )


__all__ = [
    "CrossSectionResult",
    "calculate_total_cross_sections",
    "calculate_relative_cross_sections",
    "RelativeCrossSectionResult",
    "calculate_relative_cross_sections_from_total",
]
