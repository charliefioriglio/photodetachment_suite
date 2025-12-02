"""Factory for physical dipole continuum functions."""

from __future__ import annotations

import functools
from typing import Any, Mapping

import numpy as np

from .wavefunction import (
    PhysicalDipoleParameters,
    build_directional_wavefunction,
    build_wavefunction_in_body_frame,
    prepare_body_frame,
)


def _extract_geometry(metadata: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    """Return midpoint, bond axis, and half-length derived from metadata."""

    centers = np.asarray(metadata.get("atom_centers_bohr"), dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("physical_dipole continuum requires atomic centers metadata")
    midpoint = np.asarray(metadata.get("bond_midpoint_bohr"))
    if midpoint is None:
        midpoint = centers.mean(axis=0)
    axis = metadata.get("bond_axis")
    if axis is None and centers.shape[0] == 2:
        axis = centers[0] - centers[1]
    if axis is None:
        raise ValueError("Unable to determine bond axis for physical dipole continuum")
    axis = np.asarray(axis, dtype=float)
    half_length = metadata.get("half_bond_length_bohr")
    if half_length is None and centers.shape[0] == 2:
        half_length = 0.5 * float(np.linalg.norm(centers[0] - centers[1]))
    if half_length is None or half_length <= 0.0:
        raise ValueError("physical_dipole continuum requires non-zero bond length")
    return midpoint, axis, float(half_length)


def make_physical_dipole_continuum(
    *,
    dipole_strength: float,
    orbital_metadata: Mapping[str, Any],
    l_max: int = 20,
    n_mode: int = 0,
    m_quantum: int = 0,
    xi_match: float = 2.5,
    radial_max_terms: int = 60,
    sum_modes: bool = False,
    max_modes_per_m: int | None = None,
    max_m: int | None = None,
) -> callable:
    """Return a continuum callable implementing the physical dipole model."""

    midpoint, bond_axis, half_length = _extract_geometry(orbital_metadata)

    params = PhysicalDipoleParameters(
        dipole_strength=float(dipole_strength),
        half_bond_length_bohr=half_length,
        l_max=int(l_max),
        n_mode=int(n_mode),
        m_quantum=int(m_quantum),
        xi_match=float(xi_match),
        radial_max_terms=int(radial_max_terms),
    )

    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    mode_cache: dict[int, dict[tuple[int, int, float], np.ndarray]] = {}
    angular_cache: dict[
        tuple[int, float, float, float, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}

    sum_modes = bool(sum_modes)

    def continuum(k_vec: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        k_vec = np.asarray(k_vec, dtype=float)
        k_mag = float(np.linalg.norm(k_vec))
        if k_mag == 0.0:
            return np.zeros_like(X, dtype=np.complex128)

        energy_au = 0.5 * k_mag**2
        grid_key = id(X)
        if grid_key not in cache:
            cache[grid_key] = prepare_body_frame(
                X,
                Y,
                Z,
                midpoint=midpoint,
                bond_axis=bond_axis,
            )
        X_body, Y_body, Z_body = cache[grid_key]

        if sum_modes:
            grid_mode_cache = mode_cache.setdefault(grid_key, {})
            return build_directional_wavefunction(
                params,
                energy_au,
                X_body,
                Y_body,
                Z_body,
                k_vec,
                mode_cache=grid_mode_cache,
                angular_cache=angular_cache,
                max_modes_per_m=max_modes_per_m,
                max_m=max_m,
            )

        return build_wavefunction_in_body_frame(params, energy_au, X_body, Y_body, Z_body)

    return continuum


__all__ = ["make_physical_dipole_continuum"]
