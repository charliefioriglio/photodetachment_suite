from __future__ import annotations

"""Utility to evaluate spherical Bessel functions for fractional order."""

import numpy as np
from scipy.special import jv, yv


def spherical_bessel_general_order(order: float, kr: np.ndarray) -> np.ndarray:
    kr = np.asarray(kr, dtype=float)
    kr_safe = np.maximum(kr, 1.0e-12)
    nu = order + 0.5

    prefactor = np.sqrt(np.pi / (2.0 * kr_safe))
    j_part = prefactor * jv(nu, kr_safe)

    if order >= 0:
        return j_part

    order_abs = -order
    y_part = prefactor * yv(order_abs + 0.5, kr_safe)
    return np.cos(order_abs * np.pi) * j_part - np.sin(order_abs * np.pi) * y_part


__all__ = ["spherical_bessel_general_order"]
