from __future__ import annotations

"""Solve for the radial parameter ``v`` in the physical dipole expansion."""

import math
from typing import Callable, Iterable, List

import numpy as np
from scipy.optimize import brentq, fsolve

from .recurence_terms import alpha_L, beta_L, gamma_L


def _continued_fraction(
    indices: Iterable[int],
    alpha_cb: Callable[[int], complex],
    beta_cb: Callable[[int], complex],
    gamma_cb: Callable[[int], complex],
) -> complex:
    value = 0.0
    first = True
    for idx in indices:
        beta_val = beta_cb(idx)
        if first:
            value = beta_val
            first = False
            continue

        alpha_val = alpha_cb(idx)
        gamma_val = gamma_cb(idx)
        if abs(value) < 1.0e-15:
            value = beta_val
        else:
            value = beta_val - alpha_val * gamma_val / value
    return value


def _left_fraction(c: float, m: int, v: complex, Alm: float, max_terms: int) -> complex:
    indices = [-2 * i for i in range(1, max_terms + 1)]
    cf = _continued_fraction(
        indices,
        lambda L: alpha_L(L - 2, c, m, v),
        lambda L: beta_L(L, c, m, v, Alm),
        lambda L: gamma_L(L, c, m, v),
    )
    alpha_neg2 = alpha_L(-2, c, m, v)
    gamma_0 = gamma_L(0, c, m, v)
    if abs(cf) < 1.0e-15:
        return np.inf
    return alpha_neg2 * gamma_0 / cf


def _right_fraction(c: float, m: int, v: complex, Alm: float, max_terms: int) -> complex:
    indices = [2 * i for i in range(1, max_terms + 1)]
    cf = _continued_fraction(
        indices,
        lambda L: alpha_L(L, c, m, v),
        lambda L: beta_L(L, c, m, v, Alm),
        lambda L: gamma_L(L + 2, c, m, v),
    )
    alpha_0 = alpha_L(0, c, m, v)
    gamma_2 = gamma_L(2, c, m, v)
    if abs(cf) < 1.0e-15:
        return np.inf
    return alpha_0 * gamma_2 / cf


def _residual(v: complex, c: float, m: int, Alm: float, max_terms: int) -> complex:
    beta0 = beta_L(0, c, m, v, Alm)
    left = _left_fraction(c, m, v, Alm, max_terms)
    right = _right_fraction(c, m, v, Alm, max_terms)
    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan
    return beta0 - left - right


def _initial_guess(c: float, m: int, Alm: float) -> float:
    if Alm >= 0.0:
        l_guess = 0.5 * (-1.0 + math.sqrt(1.0 + 4.0 * Alm))
    else:
        l_guess = abs(m)
    v_guess = l_guess - abs(m)
    return max(v_guess, 0.0)


def solve_for_v(
    c: float,
    m: int,
    Alm: float,
    *,
    max_terms: int = 50,
    verbose: bool = False,
) -> complex:
    """Solve the Leaver continued-fraction equation for the parameter ``v``."""

    v0 = _initial_guess(c, m, Alm)

    def equation(v_val: float) -> float:
        res = _residual(v_val, c, m, Alm, max_terms)
        if np.isnan(res):
            return np.inf
        return float(np.real(res))

    solutions: List[complex] = []

    try:
        root = fsolve(equation, v0, xtol=1.0e-10)[0]
        solutions.append(root)
    except Exception:
        if verbose:
            print("fsolve failed for initial guess", v0)

    if not solutions:
        for delta in np.linspace(-2.0, 2.0, 9):
            guess = max(v0 + delta, 0.0)
            try:
                root = fsolve(equation, guess, xtol=1.0e-10)[0]
                solutions.append(root)
                break
            except Exception:
                continue

    if not solutions:
        bracket = (max(v0 - 5.0, 0.0), v0 + 5.0)
        try:
            root = brentq(equation, *bracket, maxiter=100)
            solutions.append(root)
        except Exception:
            if verbose:
                print("Brent solver failed in bracket", bracket)

    if not solutions:
        raise RuntimeError("Unable to solve for v in physical dipole continuum")

    best = min(solutions, key=lambda val: abs(_residual(val, c, m, Alm, max_terms)))
    return complex(best)


__all__ = ["solve_for_v"]
