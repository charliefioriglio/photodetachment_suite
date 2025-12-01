"""Continuum wavefunction models."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Literal, Protocol

import numpy as np

from .physical_dipole import make_physical_dipole_continuum

ContinuumType = Literal["analytic", "plane_wave", "expansion", "point_dipole", "physical_dipole"]


class ContinuumFunction(Protocol):
    def __call__(self, k_vec: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:  # pragma: no cover - structural type
        ...


def _plane_wave(k_vec: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Return the analytic plane wave ``exp(i k·r)``."""

    return np.exp(1j * (k_vec[0] * X + k_vec[1] * Y + k_vec[2] * Z))


try:  # pragma: no cover - optional dependency
    from scipy.special import spherical_jn, jv, sph_harm
    from sympy.physics.wigner import wigner_3j
except ImportError:  # pragma: no cover - optional dependency
    sph_harm = None
    spherical_jn = None
    jv = None
    wigner_3j = None


def _make_plane_wave_expansion(l_max: int) -> ContinuumFunction:
    """Return a plane-wave expansion truncated at angular momentum ``l_max``."""

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


def _spherical_bessel_general_order(order: float, kr: np.ndarray) -> np.ndarray:
    """Evaluate a spherical Bessel function of (possibly fractional) order."""

    kr = np.asarray(kr, dtype=float)
    kr_safe = np.where(kr > 1.0e-12, kr, 1.0e-12)
    values = np.sqrt(np.pi / (2.0 * kr_safe)) * jv(order + 0.5, kr_safe)
    return np.where(kr > 1.0e-12, values, 0.0)


@lru_cache(maxsize=None)
def _point_dipole_eigensystem(dipole_strength: float, lam: int, l_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eigenvalues/vectors of the point-dipole angular Hamiltonian."""

    if wigner_3j is None:
        raise ImportError("SciPy is required for the point dipole continuum model")

    abs_lam = abs(lam)
    l_values = np.arange(abs_lam, l_max + 1, dtype=int)
    n_l = len(l_values)
    h_mat = np.zeros((n_l, n_l), dtype=float)

    for idx, l in enumerate(l_values):
        h_mat[idx, idx] = l * (l + 1)
        for delta in (-1, 1):
            lp = l + delta
            if lp < abs_lam or lp > l_max:
                continue
            j = idx + delta
            coeff = -2.0 * dipole_strength * math.sqrt((2 * l + 1) * (2 * lp + 1))
            w3j = float(wigner_3j(lp, 1, l, 0, 0, 0)) * float(wigner_3j(lp, 1, l, -lam, 0, lam))
            h_mat[idx, j] += coeff * w3j
            h_mat[j, idx] = h_mat[idx, j]

    eigvals, eigvecs = np.linalg.eigh(h_mat)
    return eigvals, eigvecs, l_values


def _evaluate_omega_fields(
    eigvecs: np.ndarray,
    l_values: np.ndarray,
    lam: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Project spherical harmonics onto the dipole eigenbasis."""

    harmonics = np.array([sph_harm(lam, l, phi, theta) for l in l_values])
    return np.tensordot(eigvecs.T, harmonics, axes=([1], [0]))


def _point_dipole_coefficient(
    eigvals: np.ndarray,
    omega_dir: np.ndarray,
    mode_idx: int,
) -> complex:
    """Return the continuum expansion coefficient for a single eigenmode."""

    eigval = eigvals[mode_idx]
    L_eff = 0.5 * (-1.0 + math.sqrt(1.0 + 4.0 * eigval))
    phase = np.exp(0.5j * math.pi * L_eff)
    return 4.0 * math.pi * phase * np.conjugate(omega_dir[mode_idx])


def _evaluate_point_dipole_angular(
    dipole_strength: float,
    lam: int,
    l_max: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate point-dipole angular eigenmodes on the supplied grid."""

    if sph_harm is None:
        raise ImportError("SciPy is required for the point dipole continuum model")

    eigvals, eigvecs, l_values = _point_dipole_eigensystem(dipole_strength, lam, l_max)
    omega = _evaluate_omega_fields(eigvecs, l_values, lam, theta, phi)
    return eigvals, eigvecs, l_values, omega


def _cartesian_to_spherical(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(r, θ, φ)`` computed from Cartesian coordinates."""

    r = np.sqrt(X * X + Y * Y + Z * Z)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.divide(Z, r, out=np.zeros_like(Z), where=r > 0.0)
    theta = np.where(r > 0.0, np.arccos(np.clip(cos_theta, -1.0, 1.0)), 0.0)
    phi = np.mod(np.arctan2(Y, X), 2.0 * math.pi)
    return r, theta, phi


def _make_point_dipole_continuum(dipole_strength: float, l_max: int) -> ContinuumFunction:
    """Construct the anisotropic point-dipole continuum model."""

    if wigner_3j is None or sph_harm is None or jv is None:
        raise ImportError("SciPy is required for the point dipole continuum model")
    if l_max < 0:
        raise ValueError("l_max must be non-negative")

    D = float(dipole_strength)
    if abs(D) < 1.0e-12:
        return _plane_wave

    l_max = int(l_max)

    def continuum(k_vec: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        k_vec = np.asarray(k_vec, dtype=float)
        k_mag = float(np.linalg.norm(k_vec))
        if k_mag == 0.0:
            return np.ones_like(X, dtype=np.complex128)

        r, theta, phi = _cartesian_to_spherical(X, Y, Z)

        k_hat = k_vec / k_mag
        theta_k = math.acos(np.clip(k_hat[2], -1.0, 1.0)) if k_mag > 0.0 else 0.0
        phi_k = (math.atan2(k_hat[1], k_hat[0]) + 2.0 * math.pi) % (2.0 * math.pi) if k_mag > 0.0 else 0.0
        theta_dir = np.array([theta_k])
        phi_dir = np.array([phi_k])

        psi = np.zeros_like(X, dtype=np.complex128)
        for lam in range(-l_max, l_max + 1):
            eigvals, eigvecs, l_values, omega_grid = _evaluate_point_dipole_angular(D, lam, l_max, theta, phi)
            omega_dir = _evaluate_omega_fields(eigvecs, l_values, lam, theta_dir, phi_dir)
            omega_dir = np.squeeze(omega_dir, axis=-1)

            for mode_idx, eigval in enumerate(eigvals):
                if eigval < -0.25 or not np.isfinite(eigval):
                    continue
                L_eff = 0.5 * (-1.0 + math.sqrt(1.0 + 4.0 * eigval))
                if not np.isfinite(L_eff):
                    continue
                radial = _spherical_bessel_general_order(L_eff, k_mag * r)
                mode = radial * omega_grid[mode_idx]
                coeff = _point_dipole_coefficient(eigvals, omega_dir, mode_idx)
                psi += coeff * mode

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
        truncation level. The ``"point_dipole"`` model accepts ``dipole_strength``
        (atomic units, default ``0.0``) and an optional ``l_max`` controlling the
        angular basis size.
    """

    if kind in _AVAILABLE:
        return _AVAILABLE[kind]
    if kind == "expansion":
        l_max = kwargs.get("l_max")
        if l_max is None:
            raise ValueError("plane-wave expansion requires l_max")
        return _make_plane_wave_expansion(int(l_max))
    if kind == "point_dipole":
        dipole_strength = float(kwargs.get("dipole_strength", 0.0))
        l_max = int(kwargs.get("l_max", 6))
        return _make_point_dipole_continuum(dipole_strength, l_max)
    if kind == "physical_dipole":
        if "orbital_metadata" not in kwargs:
            raise ValueError("physical_dipole continuum requires orbital_metadata")
        return make_physical_dipole_continuum(**kwargs)
    raise NotImplementedError(f"Continuum model {kind!r} is not implemented yet")


__all__ = ["ContinuumFunction", "ContinuumType", "get_continuum_function"]
