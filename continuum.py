"""Continuum wavefunction models."""

from __future__ import annotations

import math
from typing import Dict, Literal, Protocol

import numpy as np

ContinuumType = Literal["analytic", "plane_wave", "expansion", "point_dipole", "physical_dipole"]


class ContinuumFunction(Protocol):
    def __call__(self, k_vec: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:  # pragma: no cover - structural type
        ...


def _plane_wave(k_vec: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return np.exp(1j * (k_vec[0] * X + k_vec[1] * Y + k_vec[2] * Z))


try:  # pragma: no cover - optional dependency
    from scipy.special import sph_harm, spherical_jn
except ImportError:  # pragma: no cover - optional dependency
    sph_harm = None
    spherical_jn = None


def _make_plane_wave_expansion(l_max: int) -> ContinuumFunction:
    if l_max < 0:
        raise ValueError("l_max must be non-negative")
    if sph_harm is None or spherical_jn is None:
        raise ImportError("SciPy is required for the plane-wave expansion continuum")

    def continuum(k_vec: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        k_vec = np.asarray(k_vec, dtype=float)
        k_mag = float(np.linalg.norm(k_vec))
        if k_mag == 0.0:
            return np.ones_like(X, dtype=np.complex128)

        r = np.sqrt(X * X + Y * Y + Z * Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = np.arccos(np.clip(np.divide(Z, r, out=np.zeros_like(Z), where=r > 0.0), -1.0, 1.0))
        phi = np.arctan2(Y, X)

        k_hat = k_vec / k_mag
        theta_k = math.acos(np.clip(k_hat[2], -1.0, 1.0))
        phi_k = math.atan2(k_hat[1], k_hat[0])

        psi = np.zeros_like(X, dtype=np.complex128)
        four_pi = 4.0 * math.pi

        for l in range(l_max + 1):
            jl = spherical_jn(l, k_mag * r)
            prefactor = (1j) ** l
            for m in range(-l, l + 1):
                Y_lm_r = sph_harm(l, m, theta, phi)
                Y_lm_k = np.conj(sph_harm(l, m, theta_k, phi_k))
                psi += four_pi * prefactor * jl * Y_lm_k * Y_lm_r

        return psi

    return continuum


_AVAILABLE: Dict[str, ContinuumFunction] = {
    "analytic": _plane_wave,
    "plane_wave": _plane_wave,
}


def get_continuum_function(kind: ContinuumType, /, **kwargs) -> ContinuumFunction:
    """Return the continuum wavefunction generator for *kind*.

    Parameters
    ----------
    kind:
        Continuum model identifier.
    **kwargs:
        Additional parameters for specific models. The ``"expansion"`` model
        expects an integer ``l_max`` describing the spherical Bessel/harmonic
        truncation level.
    """

    if kind in _AVAILABLE:
        return _AVAILABLE[kind]
    if kind == "expansion":
        l_max = kwargs.get("l_max")
        if l_max is None:
            raise ValueError("plane-wave expansion requires l_max")
        return _make_plane_wave_expansion(int(l_max))
    raise NotImplementedError(f"Continuum model {kind!r} is not implemented yet")


__all__ = ["ContinuumFunction", "ContinuumType", "get_continuum_function"]
