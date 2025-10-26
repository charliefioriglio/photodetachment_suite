"""Cartesian grids used to evaluate Dyson orbitals and continuum states."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CartesianGrid:
    """Uniform Cartesian grid describing the evaluation volume."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    indexing: str = "ij"

    def __post_init__(self) -> None:
        if self.indexing not in {"ij", "xy"}:
            raise ValueError("indexing must be 'ij' or 'xy'")
        for axis, name in ((self.x, "x"), (self.y, "y"), (self.z, "z")):
            if axis.ndim != 1:
                raise ValueError(f"{name}-axis must be 1D")
        if len(self.x) < 2 or len(self.y) < 2 or len(self.z) < 2:
            raise ValueError("each axis must contain at least two sample points")

    @cached_property
    def mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``np.meshgrid`` arrays with the configured indexing."""

        return np.meshgrid(self.x, self.y, self.z, indexing=self.indexing)

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Return the grid spacing assuming uniform axes."""

        dx = float(self.x[1] - self.x[0])
        dy = float(self.y[1] - self.y[0])
        dz = float(self.z[1] - self.z[0])
        return dx, dy, dz


__all__ = ["CartesianGrid"]
