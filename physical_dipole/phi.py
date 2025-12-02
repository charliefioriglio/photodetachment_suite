"""Azimuthal factors for the physical dipole continuum."""

from __future__ import annotations

import numpy as np


def compute_phi_m(m: int, phi: np.ndarray) -> np.ndarray:
    """Return the normalized azimuthal factor ``(2π)^(-1/2) e^{i m φ}``."""

    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(1j * m * phi)


__all__ = ["compute_phi_m"]
