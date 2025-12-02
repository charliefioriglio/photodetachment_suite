"""Assemble physical dipole continuum wavefunctions on Cartesian grids."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import lpmv

from .angular import solve_angular_eigenproblem
from .gallup import solve_radial_m0_with_asymptotic_matching
from .phi import compute_phi_m
from .radial import radial_function


@dataclass(frozen=True)
class PhysicalDipoleParameters:
    """Numerical configuration for constructing physical dipole continua."""

    dipole_strength: float
    half_bond_length_bohr: float
    l_max: int = 20
    n_mode: int = 0
    m_quantum: int = 0
    xi_match: float = 2.5
    n_xi: int = 200
    n_eta: int = 200
    n_phi: int = 100
    radial_max_terms: int = 60


def rotation_matrix_to_align_with_z(axis: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps ``axis`` onto the +Z direction."""

    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1.0e-12:
        return np.identity(3)
    v = axis / axis_norm
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(v, z):
        return np.identity(3)
    if np.allclose(v, -z):
        return np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    cross = np.cross(v, z)
    s = np.linalg.norm(cross)
    c = float(np.dot(v, z))
    vx = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    R = np.identity(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
    return R


def prepare_body_frame(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    midpoint: np.ndarray,
    bond_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate the grid to ``midpoint`` and rotate it so the bond axis is +Z."""

    coords = np.stack((X - midpoint[0], Y - midpoint[1], Z - midpoint[2]), axis=-1)
    R = rotation_matrix_to_align_with_z(bond_axis)
    rotated = coords @ R.T
    return rotated[..., 0], rotated[..., 1], rotated[..., 2]


def prolate_spheroidal_coordinates(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    half_bond_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates into prolate spheroidal coordinates."""

    a = half_bond_length
    Z_A = a
    Z_B = -a
    rA = np.sqrt(X**2 + Y**2 + (Z - Z_A) ** 2)
    rB = np.sqrt(X**2 + Y**2 + (Z - Z_B) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = (rA + rB) / (2.0 * a)
        eta = (rA - rB) / (2.0 * a)
    phi = np.arctan2(Y, X)
    return xi, np.clip(eta, -1.0, 1.0), phi


def build_single_mode_wavefunction(
    params: PhysicalDipoleParameters,
    energy_au: float,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """Evaluate a single (m, n) body-frame continuum mode on ``(X, Y, Z)``."""

    if params.half_bond_length_bohr <= 0.0:
        raise ValueError("half_bond_length_bohr must be positive for physical dipole continuum")

    xi, eta, phi = prolate_spheroidal_coordinates(X, Y, Z, params.half_bond_length_bohr)

    m = params.m_quantum
    l_max = params.l_max
    m_abs = abs(m)
    c = params.half_bond_length_bohr * math.sqrt(2.0 * energy_au)
    n = int(params.n_mode)

    xi_vals = np.linspace(1.0, np.max(xi) * 1.05, params.n_xi)
    eta_vals = np.linspace(-1.0, 1.0, params.n_eta)

    eigvals, eigvecs, ell_vals = solve_angular_eigenproblem(
        m_abs,
        l_max,
        energy_au,
        params.half_bond_length_bohr,
        params.dipole_strength,
    )
    if n >= eigvals.size:
        raise ValueError(f"Requested angular mode {n} but only {eigvals.size} modes available")
    v_ang = eigvecs[:, n]
    P_basis = np.array([lpmv(m_abs, ell, eta_vals) for ell in ell_vals])
    T_eta_vals = P_basis.T @ v_ang
    if m < 0:
        T_eta_vals = T_eta_vals * ((-1) ** m_abs)
    T_eta_interp = interp1d(eta_vals, T_eta_vals, kind="cubic", bounds_error=False, fill_value=0.0)

    Alm = -eigvals[n]
    if m == 0:
        R_xi_vals, _, _, _ = solve_radial_m0_with_asymptotic_matching(
            c,
            Alm,
            xi_vals,
            dipole_strength=params.dipole_strength,
            max_terms=params.radial_max_terms,
            xi_match=params.xi_match,
        )
    else:
        R_xi_vals = radial_function(
            xi_vals,
            c,
            m,
            Alm,
            l_max=l_max,
            max_terms=params.radial_max_terms,
        )
    R_xi_interp = interp1d(xi_vals, R_xi_vals, kind="cubic", bounds_error=False, fill_value=0.0)

    T_eta = T_eta_interp(eta)
    R_xi = R_xi_interp(xi)
    Phi_m = compute_phi_m(m, phi)

    return R_xi * T_eta * Phi_m


def build_wavefunction_in_body_frame(
    params: PhysicalDipoleParameters,
    energy_au: float,
    X_body: np.ndarray,
    Y_body: np.ndarray,
    Z_body: np.ndarray,
) -> np.ndarray:
    """Convenience wrapper that assumes the grid is already in the body frame."""

    return build_single_mode_wavefunction(params, energy_au, X_body, Y_body, Z_body)


def _energy_key(energy_au: float) -> float:
    """Quantize the energy for cache lookups to avoid floating-point drift."""
    return float(np.round(energy_au, decimals=12))


def _angular_cache_key(
    m: int,
    energy_key: float,
    params: PhysicalDipoleParameters,
) -> Tuple[int, float, float, float, int]:
    """Key angular solutions by quantum numbers and geometry settings."""

    return (
        m,
        energy_key,
        params.half_bond_length_bohr,
        params.dipole_strength,
        params.l_max,
    )


def _evaluate_angular_mode_at_direction(
    m: int,
    n_mode: int,
    eta_dir: float,
    phi_dir: float,
    eigvecs: np.ndarray,
    ell_vals: np.ndarray,
) -> complex:
    """Evaluate an angular eigenmode along the emission direction."""

    m_abs = abs(m)
    P_basis = np.array([lpmv(m_abs, ell, eta_dir) for ell in ell_vals], dtype=float)
    coeff = float(np.dot(P_basis, eigvecs[:, n_mode]))
    if m < 0:
        coeff *= (-1) ** m_abs
    phi_factor = compute_phi_m(m, np.array([phi_dir]))[0]
    return coeff * phi_factor


def build_directional_wavefunction(
    params: PhysicalDipoleParameters,
    energy_au: float,
    X_body: np.ndarray,
    Y_body: np.ndarray,
    Z_body: np.ndarray,
    k_vec_body: np.ndarray,
    *,
    mode_cache: Dict[Tuple[int, int, float], np.ndarray],
    angular_cache: Dict[
        Tuple[int, float, float, float, int], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    max_modes_per_m: int | None = None,
    max_m: int | None = None,
) -> np.ndarray:
    """Assemble the sum over ``m``/mode contributions for a specific direction."""
    if energy_au <= 0.0:
        return np.zeros_like(X_body, dtype=np.complex128)

    k_vec = np.asarray(k_vec_body, dtype=float)
    k_mag = float(np.linalg.norm(k_vec))
    if k_mag == 0.0:
        return np.zeros_like(X_body, dtype=np.complex128)

    k_hat = k_vec / k_mag
    theta_k = math.acos(np.clip(k_hat[2], -1.0, 1.0))
    phi_k = float(np.mod(math.atan2(k_hat[1], k_hat[0]), 2.0 * math.pi))
    eta_dir = math.cos(theta_k)

    grid_energy_key = _energy_key(energy_au)
    m_limit = params.l_max if max_m is None else min(params.l_max, int(max_m))
    max_modes_per_m = None if max_modes_per_m is None else max(1, int(max_modes_per_m))

    total = np.zeros_like(X_body, dtype=np.complex128)
    for m in range(-m_limit, m_limit + 1):
        m_abs = abs(m)
        if m_abs > params.l_max:
            continue

        eig_cache_key = _angular_cache_key(m, grid_energy_key, params)
        if eig_cache_key in angular_cache:
            eigvals, eigvecs, ell_vals = angular_cache[eig_cache_key]
        else:
            eigvals, eigvecs, ell_vals = solve_angular_eigenproblem(
                m_abs,
                params.l_max,
                energy_au,
                params.half_bond_length_bohr,
                params.dipole_strength,
            )
            angular_cache[eig_cache_key] = (eigvals, eigvecs, ell_vals)

        if eigvals.size == 0:
            continue

        max_n = params.l_max - m_abs
        if max_modes_per_m is not None:
            max_n = min(max_n, max_modes_per_m - 1)
        n_count = min(max_n + 1, eigvals.size)
        if n_count <= 0:
            continue

        for n_mode in range(n_count):
            coeff = _evaluate_angular_mode_at_direction(
                m,
                n_mode,
                eta_dir,
                phi_k,
                eigvecs,
                ell_vals,
            )
            if abs(coeff) < 1.0e-14:
                continue

            cache_key = (m, n_mode, grid_energy_key)
            if cache_key not in mode_cache:
                mode_params = replace(params, m_quantum=m, n_mode=n_mode)
                mode_cache[cache_key] = build_single_mode_wavefunction(
                    mode_params,
                    energy_au,
                    X_body,
                    Y_body,
                    Z_body,
                )
            total += coeff * mode_cache[cache_key]

    return total


__all__ = [
    "PhysicalDipoleParameters",
    "prepare_body_frame",
    "prolate_spheroidal_coordinates",
    "build_wavefunction_in_body_frame",
    "build_single_mode_wavefunction",
    "build_directional_wavefunction",
]
