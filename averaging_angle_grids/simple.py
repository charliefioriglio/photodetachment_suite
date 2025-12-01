"""Lightweight quasi-uniform Euler angle grids."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:  # pragma: no cover
    from . import AngleGrid


def _alloc_alpha_counts(weights: np.ndarray, target_total: int) -> np.ndarray:
    """Allocate how many alpha samples each beta band should receive.

    Uses weighted rounding with a simple largest-remainder correction so the
    total matches ``target_total`` while ensuring every band receives at least
    one sample. This keeps the subsequent alpha stratification numerically
    stable even for small grids.
    """

    raw = weights * target_total
    counts = np.maximum(1, np.floor(raw).astype(int))
    remainder = target_total - int(counts.sum())

    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        idx = 0
        while remainder > 0:
            counts[order[idx % len(order)]] += 1
            remainder -= 1
            idx += 1
    elif remainder < 0:
        order = np.argsort(raw - counts)
        idx = 0
        while remainder < 0:
            target = order[idx % len(order)]
            if counts[target] > 1:
                counts[target] -= 1
                remainder += 1
            idx += 1
    return counts


def generate_simple_grid(
    *,
    n_orientations: int,
    gamma_fixed: float = 0.0,
    seed: int | np.random.Generator | None = None,
) -> "AngleGrid":
    """Generate a simple stratified Euler angle grid.

    Parameters
    ----------
    n_orientations:
        Total number of (alpha, beta) samples to generate.
    gamma_fixed:
        Constant gamma angle (ZYZ convention) assigned to each sample.
    seed:
        Optional seed or ``numpy.random.Generator`` for reproducibility.

    Returns
    -------
    AngleGrid
        Euler angles (``N × 3``) with uniform quadrature weights.
    """

    if n_orientations < 1:
        raise ValueError("n_orientations must be at least 1")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    n_beta = max(1, int(round(math.sqrt(n_orientations))))
    db = math.pi / n_beta

    beta_offset = float(rng.uniform(0.0, db))
    beta_vals = np.linspace(0.0, math.pi, n_beta, endpoint=False) + beta_offset
    beta_vals = beta_vals[(beta_vals > 0.0) & (beta_vals < math.pi)]
    if beta_vals.size == 0:
        beta_vals = np.array([math.pi / 2])

    sin_betas = np.sin(beta_vals)
    weights = sin_betas / sin_betas.sum()
    alpha_counts = _alloc_alpha_counts(weights, n_orientations)

    angles: list[tuple[float, float, float]] = []
    for beta, count in zip(beta_vals, alpha_counts, strict=True):
        alpha_offset = float(rng.uniform(0.0, 2.0 * math.pi))
        alpha_vals = (
            np.linspace(0.0, 2.0 * math.pi, count, endpoint=False) + alpha_offset
        ) % (2.0 * math.pi)
        angles.extend((float(alpha), float(beta), float(gamma_fixed)) for alpha in alpha_vals)

    if len(angles) > n_orientations:
        angles = angles[:n_orientations]

    if not angles:
        angles.append((0.0, beta_vals[0], float(gamma_fixed)))

    from . import AngleGrid

    angle_array = np.array(angles, dtype=float)
    weights = np.full(angle_array.shape[0], 1.0 / angle_array.shape[0])
    return AngleGrid(angles=angle_array, weights=weights)


__all__ = ["generate_simple_grid"]
