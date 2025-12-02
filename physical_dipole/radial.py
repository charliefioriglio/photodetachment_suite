"""Evaluate the radial part of the physical dipole continuum wavefunction."""

from __future__ import annotations

import numpy as np

from .bessel import spherical_bessel_general_order
from .radial_coefficients import calculate_radial_coefficients


def radial_function(
    xi_vals: np.ndarray,
    c: float,
    m: int,
    Alm: float,
    *,
    l_max: int = 20,
    max_terms: int = 50,
) -> np.ndarray:
    """Evaluate the physical dipole radial basis for arbitrary ``m``."""

    xi = np.asarray(xi_vals)

    if Alm == 0.0 and m == 0:
        coefficients = {0: 1.0}
        v = 0.0
    else:
        coefficients, v = calculate_radial_coefficients(c, m, Alm, l_max=l_max, max_terms=max_terms)

    prefactor = np.power((xi**2 - 1.0) / xi**2, abs(m) / 2.0)
    radial_sum = np.zeros_like(xi, dtype=complex)
    kr = xi * c

    for L, coeff in coefficients.items():
        order = L + v
        radial_sum += coeff * spherical_bessel_general_order(order, kr)

    return prefactor * radial_sum


__all__ = ["radial_function"]
