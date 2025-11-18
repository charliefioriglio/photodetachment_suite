from __future__ import annotations

"""Gallup power-series solution for the m=0 physical dipole radial equation."""

import numpy as np

from .angular import solve_angular_eigenproblem

_LOW_D_DATA = np.array(
    [
        [0.05, 0.84012, 0.49352, -0.04488],
        [0.1, 0.83982, 0.47380, -0.03613],
        [0.15, 0.83673, 0.43988, -0.00721],
        [0.2, 0.82620, 0.38983, 0.07240],
        [0.25, 0.80013, 0.32001, 0.26619],
        [0.3, 0.74529, 0.22376, 0.69806],
        [0.319, 0.81129, 0.17796, 1.09624],
    ]
)

_HIGH_D_DATA = np.array(
    [
        [0.35, 0.93442, 1.25271, 0.13393, 1.40636],
        [0.375, 0.87407, 0.83232, 0.18837, 1.25383],
        [0.4, 0.99414, 0.49319, 0.26820, 1.29025],
        [0.425, 1.05239, 0.30730, 0.38376, 1.46937],
        [0.45, 1.03035, 0.23666, 0.48631, 1.61464],
        [0.475, 1.00443, 0.19493, 0.55311, 1.76137],
        [0.5, 0.98588, 0.16401, 0.60814, 1.46454],
        [0.55, 0.97406, 0.12825, 0.67155, 1.01760],
        [0.6, 0.96821, 0.10357, 0.72504, 0.63462],
        [0.65, 0.96205, 0.08215, 0.78544, 0.34899],
        [0.7, 0.95745, 0.06588, 0.84898, 0.10428],
        [0.75, 0.95473, 0.05367, 0.91290, -0.10766],
        [0.8, 0.95338, 0.04415, 0.97988, -0.27209],
        [0.85, 0.95234, 0.03638, 1.03991, -0.46058],
        [0.9, 0.95156, 0.03028, 1.09754, -0.65150],
        [0.95, 0.95105, 0.02546, 1.15262, -0.83972],
        [1.0, 0.95073, 0.02145, 1.20573, -1.02719],
        [1.05, 0.95050, 0.01820, 1.25875, -1.20609],
        [1.1, 0.95034, 0.01557, 1.31004, -1.38346],
        [1.15, 0.95021, 0.01331, 1.35951, -1.56172],
        [1.2, 0.95014, 0.01145, 1.40784, -1.74021],
        [1.25, 0.95006, 0.00986, 1.45548, -1.91368],
        [2.3, 0.95004, 0.00857, 1.50203, -2.08278],
    ]
)


def get_gallup_normalization_a0(dipole_strength: float, c: float) -> float:
    if dipole_strength <= 0.32:
        data = _LOW_D_DATA
        if dipole_strength <= data[0, 0]:
            a, x, b = data[0, 1:4]
        elif dipole_strength >= data[-1, 0]:
            a, x, b = data[-1, 1:4]
        else:
            a = np.interp(dipole_strength, data[:, 0], data[:, 1])
            x = np.interp(dipole_strength, data[:, 0], data[:, 2])
            b = np.interp(dipole_strength, data[:, 0], data[:, 3])
        return a * (c**x) * (1.0 + b * c**2)

    data = _HIGH_D_DATA
    if dipole_strength <= data[0, 0]:
        a, b, g, d = data[0, 1:5]
    elif dipole_strength >= data[-1, 0]:
        a, b, g, d = data[-1, 1:5]
    else:
        a = np.interp(dipole_strength, data[:, 0], data[:, 1])
        b = np.interp(dipole_strength, data[:, 0], data[:, 2])
        g = np.interp(dipole_strength, data[:, 0], data[:, 3])
        d = np.interp(dipole_strength, data[:, 0], data[:, 4])
    if c <= 0.0:
        return 1.0
    return 1.0 / (a + b * np.cos(2.0 * g * np.log(c) + d))


def solve_radial_m0_gallup_method(
    c: float,
    A00: float,
    xi_vals: np.ndarray,
    *,
    dipole_strength: float | None = None,
    max_terms: int = 50,
    use_gallup_a0: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if use_gallup_a0 and dipole_strength is not None:
        a0 = get_gallup_normalization_a0(dipole_strength, c)
    else:
        a0 = 1.0

    coeffs = np.zeros(max_terms, dtype=complex)
    coeffs[0] = a0
    c2 = c**2
    coeffs[1] = -0.5 * (c2 - A00) * coeffs[0]

    for k in range(2, max_terms):
        coeff_km1 = c2 - A00 + k * (k - 1)
        coeff_km2 = 2.0 * c2
        coeff_km3 = c2 if k >= 3 else 0.0
        numerator = coeff_km1 * coeffs[k - 1] + coeff_km2 * coeffs[k - 2]
        if k >= 3:
            numerator += coeff_km3 * coeffs[k - 3]
        coeffs[k] = -numerator / (2.0 * k * k)

    xi_vals = np.asarray(xi_vals)
    R = np.zeros_like(xi_vals, dtype=complex)
    for idx, xi in enumerate(xi_vals):
        if abs(xi - 1.0) < 1.0e-12:
            R[idx] = coeffs[0]
            continue
        xi_minus_1 = xi - 1.0
        partial = 0.0
        for k, a_k in enumerate(coeffs):
            if abs(a_k) < 1.0e-16:
                break
            term = a_k * (xi_minus_1**k)
            partial += term
            if k > 5 and abs(term) < 1.0e-15 * abs(partial):
                break
        R[idx] = partial
    return R, coeffs


def solve_radial_m0_with_asymptotic_matching(
    c: float,
    A00: float,
    xi_vals: np.ndarray,
    *,
    dipole_strength: float | None,
    max_terms: int = 50,
    xi_match: float = 2.5,
    use_gallup_a0: bool = True,
) -> tuple[np.ndarray, np.ndarray, float | complex, float]:
    xi_vals = np.asarray(xi_vals)
    R_gallup, coeffs = solve_radial_m0_gallup_method(
        c,
        A00,
        xi_vals,
        dipole_strength=dipole_strength,
        max_terms=max_terms,
        use_gallup_a0=use_gallup_a0,
    )

    if use_gallup_a0 and dipole_strength is not None:
        R_gallup = R_gallup / get_gallup_normalization_a0(0.0, c)

    match_idx = int(np.argmin(np.abs(xi_vals - xi_match)))
    xi_match_actual = xi_vals[match_idx]
    R_match = R_gallup[match_idx]

    if 0 < match_idx < len(xi_vals) - 1:
        dxi = xi_vals[match_idx + 1] - xi_vals[match_idx - 1]
        dR_match = (R_gallup[match_idx + 1] - R_gallup[match_idx - 1]) / dxi
    elif match_idx == 0:
        dxi = xi_vals[1] - xi_vals[0]
        dR_match = (R_gallup[1] - R_gallup[0]) / dxi
    else:
        dxi = xi_vals[-1] - xi_vals[-2]
        dR_match = (R_gallup[-1] - R_gallup[-2]) / dxi

    sqrt_term = np.sqrt(xi_match_actual**2 - 1.0)
    best_delta = 0.0
    best_error = float("inf")
    best_C = 0.0
    for delta_trial in np.linspace(-np.pi, np.pi, 400):
        sin_term = np.sin(c * xi_match_actual + delta_trial)
        if abs(sin_term) < 1.0e-12:
            continue
        C_candidate = R_match * sqrt_term / sin_term
        cos_term = np.cos(c * xi_match_actual + delta_trial)
        term1 = c * cos_term / sqrt_term
        term2 = xi_match_actual * sin_term / (xi_match_actual**2 - 1.0) ** 1.5
        dR_expected = C_candidate * (term1 - term2)
        error = abs(dR_expected - dR_match)
        if error < best_error:
            best_error = error
            best_delta = delta_trial
            best_C = C_candidate

    R = np.zeros_like(xi_vals, dtype=complex)
    for idx, xi in enumerate(xi_vals):
        if xi <= xi_match_actual or xi <= 1.0:
            R[idx] = R_gallup[idx]
        else:
            sqrt_term = np.sqrt(xi**2 - 1.0)
            R[idx] = best_C * np.sin(c * xi + best_delta) / sqrt_term
    C_value = complex(best_C)
    if abs(C_value.imag) <= 1.0e-12:
        C_out: float | complex = float(C_value.real)
    else:
        C_out = C_value
    return R, coeffs, C_out, float(best_delta)


__all__ = [
    "get_gallup_normalization_a0",
    "solve_radial_m0_gallup_method",
    "solve_radial_m0_with_asymptotic_matching",
]
