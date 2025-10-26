"""Utilities for building Euler angle grids used in orientational averaging.

This package centralizes the existing angle-grid recipes (hard-coded, simple, and
repulsion-based) behind a small convenience dispatcher so downstream code can
request a grid by name without needing to know the implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Protocol

import numpy as np

from .hard_coded import load_hard_coded_grid
from .repulsion import generate_repulsion_grid
from .simple import generate_simple_grid

AngleGridMethod = Literal["hard-coded", "simple", "repulsion"]


class GridFactory(Protocol):
    def __call__(self, *args, **kwargs) -> "AngleGrid":  # pragma: no cover - typing protocol
        ...


@dataclass(frozen=True)
class AngleGrid:
    """Container for Euler angles and their quadrature weights."""

    angles: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.angles.ndim != 2 or self.angles.shape[1] != 3:
            raise ValueError("angles must be a 2D array with shape (N, 3)")
        if self.weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if self.angles.shape[0] != self.weights.shape[0]:
            raise ValueError("angles and weights must share the same length")

    def as_matrix(self) -> np.ndarray:
        """Return an array with shape ``(4, N)`` = (alpha, beta, gamma, weights)."""

        return np.vstack((self.angles.T, self.weights))


_FACTORIES: Dict[AngleGridMethod, GridFactory] = {
    "hard-coded": load_hard_coded_grid,
    "simple": generate_simple_grid,
    "repulsion": generate_repulsion_grid,
}


def available_methods() -> Iterable[AngleGridMethod]:
    """Return an iterable of supported grid method names."""

    return _FACTORIES.keys()


def get_angle_grid(method: AngleGridMethod, /, **kwargs) -> AngleGrid:
    """Return an :class:`AngleGrid` instance for the selected *method*.

    Parameters
    ----------
    method:
        One of ``"hard-coded"``, ``"simple"``, or ``"repulsion"``.
    **kwargs:
        Forwarded to the underlying factory. Refer to the docstring of the
        corresponding ``generate_*`` function for the accepted keyword
        arguments.

    Returns
    -------
    AngleGrid
        Euler angles and quadrature weights.
    """

    try:
        factory = _FACTORIES[method]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown angle grid method: {method!r}") from exc
    return factory(**kwargs)


__all__ = [
    "AngleGrid",
    "AngleGridMethod",
    "available_methods",
    "get_angle_grid",
    "generate_repulsion_grid",
    "generate_simple_grid",
    "load_hard_coded_grid",
]
