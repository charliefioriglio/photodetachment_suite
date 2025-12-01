"""Repulsion-optimized Euler angle grids."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:  # pragma: no cover
    from . import AngleGrid


def _fibonacci_sphere(
    n_points: int,
    *,
    rng: np.random.Generator,
    perturb_strength: float = 0.1,
) -> np.ndarray:
    """Return points quasi-uniformly distributed on a sphere via Fibonacci sampling."""

    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    i = np.arange(n_points, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / n_points
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = (2.0 * math.pi * i / golden_ratio) % (2.0 * math.pi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)

    points = np.stack((x, y, z), axis=1)

    if perturb_strength == 0.0:
        return points

    perturbed = np.empty_like(points)
    for idx, p in enumerate(points):
        rand = rng.normal(size=3)
        rand -= np.dot(rand, p) * p
        norm = np.linalg.norm(rand)
        if norm == 0.0:
            perturbed[idx] = p
            continue
        rand /= norm
        candidate = p + perturb_strength * rand
        perturbed[idx] = candidate / np.linalg.norm(candidate)
    return perturbed


def generate_repulsion_grid(
    *,
    n_orientations: int,
    gamma_fixed: float = 0.0,
    step_size: float = 1e-3,
    tol: float = 1e-4,
    max_iter: int = 5000,
    perturb_strength: float = 0.05,
    seed: int | np.random.Generator | None = None,
    verbose: bool = False,
) -> "AngleGrid":
    """Generate orientations by iteratively relaxing a repulsive point set.

    Parameters
    ----------
    n_orientations:
        Number of orientations to produce; also determines the uniform weights.
    gamma_fixed:
        Constant gamma rotation (ZYZ convention) assigned to each sample.
    step_size:
        Scalar step applied to the pairwise repulsion force.
    tol:
        Termination criterion based on the largest displacement magnitude.
    max_iter:
        Maximum number of relaxation sweeps before giving up.
    perturb_strength:
        Amplitude of the random tangential jitter applied to the initial
        Fibonacci sphere.
    seed:
        Optional integer or ``numpy.random.Generator`` used to seed the RNG.
    verbose:
        When ``True`` prints convergence progress.

    Returns
    -------
    AngleGrid
        Euler angles (``N × 3``) with uniform quadrature weights.
    """

    if n_orientations < 2:
        raise ValueError("n_orientations must be at least 2 for the repulsion grid")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    orientations = _fibonacci_sphere(
        n_orientations,
        rng=rng,
        perturb_strength=perturb_strength,
    )

    converged = False
    for iteration in range(1, max_iter + 1):
        displacements = np.zeros_like(orientations)
        max_disp = 0.0

        for i in range(n_orientations - 1):
            vi = orientations[i]
            for j in range(i + 1, n_orientations):
                vj = orientations[j]
                dot = float(np.clip(np.dot(vi, vj), -1.0, 1.0))
                theta = math.acos(dot)
                if theta == 0.0:
                    continue

                V = np.cross(vj, vi)
                dP_i = np.cross(V, vi)
                dP_j = np.cross(vj, V)

                dP_i /= np.linalg.norm(dP_i)
                dP_j /= np.linalg.norm(dP_j)

                force = step_size / (theta * theta)
                displacements[i] += force * dP_i
                displacements[j] += force * dP_j

        for idx in range(n_orientations):
            if np.allclose(displacements[idx], 0.0):
                continue
            orientations[idx] += displacements[idx]
            orientations[idx] /= np.linalg.norm(orientations[idx])
            disp_mag = float(np.linalg.norm(displacements[idx]))
            if disp_mag > max_disp:
                max_disp = disp_mag

        if max_disp < tol:
            converged = True
            if verbose:
                print(f"Repulsion grid converged after {iteration} iterations.")
            break

    if not converged and verbose:
        print(f"Repulsion grid reached max_iter={max_iter} without convergence.")

    alpha = (np.arctan2(orientations[:, 1], orientations[:, 0]) + 2.0 * math.pi) % (
        2.0 * math.pi
    )
    beta = np.arccos(np.clip(orientations[:, 2], -1.0, 1.0))
    gamma = np.full_like(alpha, float(gamma_fixed))

    from . import AngleGrid

    angle_array = np.column_stack((alpha, beta, gamma))
    weights = np.full(angle_array.shape[0], 1.0 / angle_array.shape[0])
    return AngleGrid(angles=angle_array, weights=weights)


__all__ = ["generate_repulsion_grid"]
