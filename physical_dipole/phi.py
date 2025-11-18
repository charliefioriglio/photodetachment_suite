from __future__ import annotations

"""Azimuthal factors for physical dipole continuum."""

import numpy as np


def compute_phi_m(m: int, phi: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(1j * m * phi)


__all__ = ["compute_phi_m"]
