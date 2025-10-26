"""Rotation utilities."""

from __future__ import annotations

import numpy as np


def rotation_matrix(alpha: float, beta: float, gamma: float, *, convention: str = "zyz") -> np.ndarray:
    """Return the Euler rotation matrix for the requested angles."""

    conv = convention.lower()
    if conv != "zyz":  # pragma: no cover - future extension hook
        raise NotImplementedError(f"Unsupported convention: {convention!r}")

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    rz1 = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rz2 = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])

    return rz1 @ ry @ rz2


__all__ = ["rotation_matrix"]
