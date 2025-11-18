from __future__ import annotations

"""Convenience wrappers to evaluate cross sections with the physical dipole continuum."""

from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import numpy as np

from ..averaging_angle_grids import AngleGrid
from ..grid import CartesianGrid
from ..integration import IntegrationMethod
from ..models import TransitionChannel

if TYPE_CHECKING:  # pragma: no cover
    from ..cross_sections import CrossSectionResult


def calculate_physical_dipole_cross_sections(
    channels: Sequence[TransitionChannel],
    grid: CartesianGrid,
    angle_grid: AngleGrid,
    photon_energies_ev: Iterable[float],
    *,
    dipole_strength: float,
    continuum_options: Mapping[str, object] | None = None,
    integration_method: IntegrationMethod = "trapezoidal",
    polarization_lab: np.ndarray | None = None,
    k_parallel_lab: np.ndarray | None = None,
    k_perpendicular_lab: Sequence[np.ndarray] | None = None,
) -> "CrossSectionResult":
    from ..cross_sections import calculate_total_cross_sections
    options = dict(continuum_options or {})
    options.setdefault("dipole_strength", dipole_strength)
    return calculate_total_cross_sections(
        channels=channels,
        grid=grid,
        angle_grid=angle_grid,
        photon_energies_ev=photon_energies_ev,
        continuum="physical_dipole",
        continuum_options=options,
        integration_method=integration_method,
        polarization_lab=polarization_lab,
        k_parallel_lab=k_parallel_lab,
        k_perpendicular_lab=k_perpendicular_lab,
    )


__all__ = ["calculate_physical_dipole_cross_sections"]
