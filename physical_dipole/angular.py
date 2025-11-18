from __future__ import annotations

"""Angular eigenproblem for the physical dipole continuum."""

import numpy as np
from scipy.linalg import eigh
from scipy.special import factorial


def solve_angular_eigenproblem(
    m: int,
    l_max: int,
    energy_au: float,
    half_bond_length_bohr: float,
    dipole_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the angular Sturm–Liouville problem for a physical dipole.

    Parameters
    ----------
    m:
        Magnetic quantum number.
    l_max:
        Maximum angular momentum included in the truncated basis.
    energy_au:
        Electron kinetic energy in Hartree.
    half_bond_length_bohr:
        ``a`` — half the internuclear separation in Bohr.
    dipole_strength:
        Physical dipole strength ``D`` in atomic units.

    Returns
    -------
    eigenvalues, eigenvectors, ell_values
        Arrays containing the separation constants, eigenvectors and the
        ``\ell`` values associated with each basis function.
    """

    m_abs = abs(m)
    ell_vals = np.arange(m_abs, l_max + 1, dtype=int)
    n_basis = len(ell_vals)

    c = np.sqrt(2.0 * energy_au) * half_bond_length_bohr
    H = np.zeros((n_basis, n_basis), dtype=float)
    S_diag = np.array(
        [
            2.0
            * factorial(l + m_abs)
            / ((2 * l + 1) * factorial(l - m_abs))
            for l in ell_vals
        ]
    )
    S = np.diag(S_diag)

    for i, l in enumerate(ell_vals):
        # diagonal term
        s_val = S_diag[i]
        if l - m_abs >= 0:
            f1 = -(l * (l + 1)) * s_val
            f2 = -c**2 * (
                (l + m_abs) * (l - m_abs) / ((2 * l + 1) * (2 * l - 1))
                + (l - m_abs + 1) * (l + m_abs + 1) / ((2 * l + 1) * (2 * l + 3))
            ) * s_val
            H[i, i] += f1 + f2

        if i + 1 < n_basis:
            f = (-2.0 * dipole_strength / (2 * l + 1)) * (l + 1 - m_abs)
            f *= 2.0 * factorial(l + m_abs + 1) / (
                (2 * l + 3) * factorial(l - m_abs + 1)
            )
            H[i + 1, i] += f
            H[i, i + 1] += f

        if i - 1 >= 0:
            f = (-2.0 * dipole_strength / (2 * l + 1)) * (l + m_abs)
            f *= 2.0 * factorial(l + m_abs - 1) / (
                (2 * l - 1) * factorial(l - m_abs - 1)
            )
            H[i - 1, i] += f
            H[i, i - 1] += f

        if i + 2 < n_basis:
            f = -c**2 * (l - m_abs + 1) * (l - m_abs + 2)
            f *= 2.0 * factorial(l + m_abs + 2) / (
                (2 * l + 1) * (2 * l + 3) * (2 * l + 5) * factorial(l - m_abs + 2)
            )
            H[i + 2, i] += f
            H[i, i + 2] += f

        if i - 2 >= 0:
            f = -c**2 * (l + m_abs) * (l + m_abs - 1)
            f *= 2.0 * factorial(l + m_abs - 2) / (
                (2 * l + 1) * (2 * l - 1) * (2 * l - 3) * factorial(l - m_abs - 2)
            )
            H[i - 2, i] += f
            H[i, i - 2] += f

    eigvals, eigvecs = eigh(H, S)
    # Sort descending to match historical convention
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    return eigvals, eigvecs, ell_vals


__all__ = ["solve_angular_eigenproblem"]
