"""Numerical integration helpers for volumetric data."""

from __future__ import annotations

from typing import Literal

import numpy as np

try:  # pragma: no cover - optional dependency
    from scipy.integrate import simpson as _scipy_simpson
except ImportError:  # pragma: no cover - optional dependency
    _scipy_simpson = None  # type: ignore[assignment]

IntegrationMethod = Literal["trapezoidal", "simpson"]


def integrate_scalar_field(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    method: IntegrationMethod = "trapezoidal",
) -> complex:
    """Integrate a scalar field defined on a tensor-product grid.

    Parameters
    ----------
    values:
        Array with shape ``(len(x), len(y), len(z))`` representing the integrand.
            Complex-valued inputs are supported and preserved in the return value.
    x, y, z:
        One-dimensional coordinate arrays describing the grid axes.
    method:
        Either ``"trapezoidal"`` (default) or ``"simpson"``. The Simpson rule
        requires SciPy ``>= 1.6`` to be available.
    """

    arr = np.asarray(values)
    if arr.shape != (len(x), len(y), len(z)):
        raise ValueError("values must have shape (len(x), len(y), len(z))")

    if method == "trapezoidal":
        tmp = np.trapz(arr, z, axis=2)
        tmp = np.trapz(tmp, y, axis=1)
        return np.trapz(tmp, x, axis=0)

    if method == "simpson":
        if _scipy_simpson is None:
            raise ImportError("SciPy is required for Simpson integration")
        tmp = _scipy_simpson(arr, z, axis=2)
        tmp = _scipy_simpson(tmp, y, axis=1)
        return _scipy_simpson(tmp, x, axis=0)

    raise ValueError(f"Unknown integration method: {method}")


__all__ = ["IntegrationMethod", "integrate_scalar_field"]
