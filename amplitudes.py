"""Shared routines for computing transition amplitudes."""

from __future__ import annotations

import numpy as np

from .averaging_angle_grids import AngleGrid
from .continuum import ContinuumType, get_continuum_function
from .grid import CartesianGrid
from .integration import IntegrationMethod, integrate_scalar_field
from .models import OrbitalPair
from .rotation import rotation_matrix


def compute_orientational_amplitudes(
    orbital: OrbitalPair,
    grid: CartesianGrid,
    angle_grid: AngleGrid,
    k_lab: np.ndarray,
    pol_lab: np.ndarray,
    *,
    continuum: ContinuumType,
    integration_method: IntegrationMethod,
    continuum_options: dict | None = None,
) -> np.ndarray:
    """Return transition amplitudes for each Euler orientation."""

    X, Y, Z = grid.mesh
    x, y, z = grid.x, grid.y, grid.z
    continuum_fn = get_continuum_function(continuum, **(continuum_options or {}))

    amplitudes = np.zeros(len(angle_grid.weights), dtype=np.complex128)
    left = orbital.scaled_left()
    right = orbital.scaled_right()

    for idx, (alpha, beta, gamma) in enumerate(angle_grid.angles):
        rot = rotation_matrix(alpha, beta, gamma)
        eps_mol = rot.T @ pol_lab
        k_mol = rot.T @ k_lab

        mu = eps_mol[0] * X + eps_mol[1] * Y + eps_mol[2] * Z
        psi_f = continuum_fn(k_mol, X, Y, Z)

        integrand_L = np.conj(left) * mu * psi_f
        integrand_R = np.conj(psi_f) * mu * right

        A_L = integrate_scalar_field(integrand_L, x, y, z, method=integration_method)
        A_R = integrate_scalar_field(integrand_R, x, y, z, method=integration_method)
        amplitudes[idx] = A_L * A_R

    return amplitudes


def weighted_intensity(amplitudes: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted average of |amplitude|^2."""

    return float(np.average(np.abs(amplitudes) ** 2, weights=weights))


__all__ = ["compute_orientational_amplitudes", "weighted_intensity"]
