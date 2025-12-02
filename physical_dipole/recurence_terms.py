"""Recurrence coefficients for the physical dipole radial expansion."""

from __future__ import annotations

import numpy as np


def _safe_division(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1.0e-15:
        return 0.0
    return numerator / denominator


def alpha_L(L: int, c: float, m: int, v: complex) -> complex:
    """Upward recurrence coefficient ``α_L`` from Leaver's formulation."""

    denom = (2 * L + 2 * v + 3) * (2 * L + 2 * v + 5)
    return -c**2 * (L + v - m + 1) * (L + v - m + 2) * _safe_division(1.0, denom)


def beta_L(L: int, c: float, m: int, v: complex, Alm: float) -> complex:
    """Diagonal recurrence coefficient ``β_L`` accounting for ``A_{lm}``."""

    denom = (2 * L + 2 * v - 1) * (2 * L + 2 * v + 3)
    if abs(denom) < 1.0e-15:
        return (L + v) * (L + v + 1) - Alm

    c_term = c**2 * (2 * ((L + v) * (L + v + 1) - m**2) - 1) / denom
    return c_term + (L + v) * (L + v + 1) - Alm


def gamma_L(L: int, c: float, m: int, v: complex) -> complex:
    """Downward recurrence coefficient ``γ_L`` from Leaver's formulation."""

    denom = (2 * L + 2 * v - 1) * (2 * L + 2 * v - 3)
    return -c**2 * (L + v + m) * (L + v + m - 1) * _safe_division(1.0, denom)


__all__ = ["alpha_L", "beta_L", "gamma_L"]
