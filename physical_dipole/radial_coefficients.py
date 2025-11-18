from __future__ import annotations

"""Compute Leaver-series coefficients for the physical dipole radial solution."""

import numpy as np

from .calculate_v import solve_for_v
from .recurence_terms import alpha_L, beta_L, gamma_L


def _continued_fraction_positive(L: int, c: float, m: int, v: complex, Alm: float, max_terms: int) -> complex:
    cf_value = 0.0
    for i in range(max_terms, 0, -1):
        L_curr = L + 2 * i
        alpha_val = alpha_L(L_curr, c, m, v)
        gamma_val = gamma_L(L_curr + 2, c, m, v)
        beta_val = beta_L(L_curr, c, m, v, Alm)
        if i == max_terms:
            cf_value = beta_val
        else:
            if abs(cf_value) < 1.0e-15:
                cf_value = beta_val
            else:
                cf_value = beta_val - alpha_val * gamma_val / cf_value
    gamma_L_val = gamma_L(L, c, m, v)
    if abs(cf_value) < 1.0e-15:
        return 0.0
    return -gamma_L_val / cf_value


def _continued_fraction_negative(L: int, c: float, m: int, v: complex, Alm: float, max_terms: int) -> complex:
    cf_value = 0.0
    for i in range(max_terms, 0, -1):
        L_curr = L - 2 * i
        alpha_val = alpha_L(L_curr - 2, c, m, v)
        gamma_val = gamma_L(L_curr, c, m, v)
        beta_val = beta_L(L_curr, c, m, v, Alm)
        if i == max_terms:
            cf_value = beta_val
        else:
            if abs(cf_value) < 1.0e-15:
                cf_value = beta_val
            else:
                cf_value = beta_val - alpha_val * gamma_val / cf_value
    alpha_L_val = alpha_L(L, c, m, v)
    if abs(cf_value) < 1.0e-15:
        return 0.0
    return -alpha_L_val / cf_value


def calculate_radial_coefficients(
    c: float,
    m: int,
    Alm: float,
    *,
    l_max: int = 20,
    max_terms: int = 50,
) -> tuple[dict[int, complex], complex]:
    """Return expansion coefficients ``a_L`` and the solved ``v`` parameter."""

    v = solve_for_v(c, m, Alm, max_terms=max_terms)
    if abs(v.imag) < 1.0e-10:
        v = float(v.real)

    if l_max % 2 != 0:
        l_max += 1

    coeffs: dict[int, complex] = {0: 1.0}

    for L in range(2, l_max + 1, 2):
        ratio = _continued_fraction_positive(L, c, m, v, Alm, max_terms)
        coeffs[L] = ratio * coeffs[L - 2]

    for L in range(-2, -l_max - 1, -2):
        ratio = _continued_fraction_negative(L, c, m, v, Alm, max_terms)
        coeffs[L] = ratio * coeffs[L + 2]

    total = sum(abs(val) for val in coeffs.values())
    if total > 0.0:
        for key in list(coeffs):
            coeffs[key] /= total

    return coeffs, v


__all__ = ["calculate_radial_coefficients"]
