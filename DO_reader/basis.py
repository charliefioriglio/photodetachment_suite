from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ANGULAR_LETTER_TO_L = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
EPS = 1.0e-14


@dataclass
class ShellSpec:
    """Minimal description of a contracted Gaussian shell."""

    angular_momentum: int
    exponents: np.ndarray  # shape = (n_primitives,)
    coefficients: np.ndarray  # shape = (n_contractions, n_primitives)
    is_pure: bool
    symbol: str
    label: str
    center_bohr: np.ndarray

    def __post_init__(self) -> None:
        if self.coefficients.ndim != 2:
            raise ValueError("Shell coefficients must be a 2D array")
        if self.exponents.ndim != 1:
            raise ValueError("Shell exponents must be a 1D array")
        if self.coefficients.shape[1] != self.exponents.size:
            raise ValueError("Number of primitives mismatches between coefficients and exponents")
        self.center_bohr = np.asarray(self.center_bohr, dtype=float)

    @property
    def n_contractions(self) -> int:
        return self.coefficients.shape[0]

    @property
    def n_primitives(self) -> int:
        return self.exponents.size


@dataclass
class AtomSpec:
    symbol: str
    index: int
    center_bohr: np.ndarray
    shells: List[ShellSpec] = field(default_factory=list)


@dataclass
class DysonInfo:
    label: str
    coefficients: np.ndarray
    transition: str | None = None
    left_norm: float | None = None
    right_norm: float | None = None

    def short_label(self) -> str:
        return self.label.lower().replace(" ", "_")


@dataclass
class QChemData:
    atoms: List[AtomSpec]
    dyson_orbitals: List[DysonInfo]
    n_basis_functions: int
    pure_map: Dict[int, bool]

    def get_dyson(self, selector: str | int) -> DysonInfo:
        if isinstance(selector, int):
            return self.dyson_orbitals[selector]
        key = selector.lower().replace(" ", "_")
        for info in self.dyson_orbitals:
            if info.short_label() == key:
                return info
        raise KeyError(f"No Dyson orbital matching selector '{selector}'")


def double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    result = 1
    while n > 1:
        result *= n
        n -= 2
    return result


def norm_cartesian_gaussian(alpha: float, lx: int, ly: int, lz: int) -> float:
    l = lx + ly + lz
    prefactor = (2.0 * alpha / np.pi) ** 0.75
    numerator = (4.0 * alpha) ** l
    denom = (
        double_factorial(2 * lx - 1)
        * double_factorial(2 * ly - 1)
        * double_factorial(2 * lz - 1)
    )
    return prefactor * np.sqrt(numerator / denom)


def gaussian_primitive(
    alpha: float,
    lx: int,
    ly: int,
    lz: int,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    center: Sequence[float],
) -> np.ndarray:
    cx, cy, cz = center
    xs = X - cx
    ys = Y - cy
    zs = Z - cz
    r2 = xs * xs + ys * ys + zs * zs
    prefactor = norm_cartesian_gaussian(alpha, lx, ly, lz)
    if lx:
        xs = xs ** lx
    else:
        xs = np.ones_like(X)
    if ly:
        ys = ys ** ly
    else:
        ys = np.ones_like(Y)
    if lz:
        zs = zs ** lz
    else:
        zs = np.ones_like(Z)
    return prefactor * xs * ys * zs * np.exp(-alpha * r2)


def cartesian_monomials(l: int) -> List[Tuple[int, int, int]]:
    combos: List[Tuple[int, int, int]] = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            lz = l - lx - ly
            combos.append((lx, ly, lz))
    return combos


PURE_COMBO_COEFFS: Dict[int, List[List[Tuple[float, Tuple[int, int, int]]]]] = {
    2: [
        [(1.0, (1, 1, 0))],
        [(1.0, (0, 1, 1))],
        [(1.0, (0, 0, 2)), (-0.5, (2, 0, 0)), (-0.5, (0, 2, 0))],
        [(1.0, (1, 0, 1))],
        [
            (np.sqrt(3.0) / 2.0, (2, 0, 0)),
            (-np.sqrt(3.0) / 2.0, (0, 2, 0)),
        ],
    ],
    3: [
        [
            (np.sqrt(5.0 / 8.0) * 3.0, (2, 1, 0)),
            (-np.sqrt(5.0 / 8.0), (0, 3, 0)),
        ],
        [(1.0, (1, 1, 1))],
        [
            (np.sqrt(3.0 / 8.0) * 4.0, (0, 1, 2)),
            (-np.sqrt(3.0 / 8.0), (0, 3, 0)),
            (-np.sqrt(3.0 / 8.0), (2, 1, 0)),
        ],
        [
            (1.0, (0, 0, 3)),
            (-1.5, (2, 0, 1)),
            (-1.5, (0, 2, 1)),
        ],
        [
            (np.sqrt(3.0 / 8.0) * 4.0, (1, 0, 2)),
            (-np.sqrt(3.0 / 8.0), (3, 0, 0)),
            (-np.sqrt(3.0 / 8.0), (1, 2, 0)),
        ],
        [
            (np.sqrt(15.0) / 2.0, (2, 0, 1)),
            (-np.sqrt(15.0) / 2.0, (0, 2, 1)),
        ],
        [
            (np.sqrt(5.0 / 8.0), (3, 0, 0)),
            (-3.0 * np.sqrt(5.0 / 8.0), (1, 2, 0)),
        ],
    ],
}


def build_primitive_bundle(
    exponents: np.ndarray,
    lx: int,
    ly: int,
    lz: int,
    center: Sequence[float],
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    dV: float,
) -> Tuple[List[np.ndarray], np.ndarray]:
    values = [
        gaussian_primitive(alpha, lx, ly, lz, X, Y, Z, center)
        for alpha in exponents
    ]
    n = len(values)
    overlap = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = float(np.sum(values[i] * values[j]) * dV)
            overlap[i, j] = val
            overlap[j, i] = val
    return values, overlap


def contract_gaussians(
    primitives: Sequence[np.ndarray], overlap: np.ndarray, coeffs: np.ndarray
) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)
    ao = np.zeros_like(primitives[0])
    for c, prim in zip(coeffs, primitives):
        ao = ao + c * prim
    norm_sq = float(coeffs @ overlap @ coeffs)
    if norm_sq < EPS:
        return ao * 0.0
    norm_const = 1.0 / np.sqrt(norm_sq)
    return norm_const * ao


def degeneracy(l: int, is_pure: bool) -> int:
    if l == 0:
        return 1
    if l == 1:
        return 3
    if is_pure:
        return 2 * l + 1
    return (l + 1) * (l + 2) // 2


def enumerate_shell_functions(
    shell: ShellSpec,
    center: Sequence[float],
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    dV: float,
) -> List[np.ndarray]:
    l = shell.angular_momentum
    combos: List[List[Tuple[float, Tuple[int, int, int]]]]
    if shell.is_pure and l >= 2:
        if l not in PURE_COMBO_COEFFS:
            raise NotImplementedError(
                f"Pure spherical functions for angular momentum l={l} are not implemented"
            )
        combos = PURE_COMBO_COEFFS[l]
        monomials: Iterable[Tuple[int, int, int]] = {
            mon for combo in combos for _, mon in combo
        }
    else:
        combos = [[(1.0, mon)] for mon in cartesian_monomials(l)]
        monomials = [combo[0][1] for combo in combos]

    primitive_cache: Dict[Tuple[int, int, int], Tuple[List[np.ndarray], np.ndarray]] = {}
    for mon in monomials:
        primitive_cache[mon] = build_primitive_bundle(
            shell.exponents, mon[0], mon[1], mon[2], center, X, Y, Z, dV
        )

    functions: List[np.ndarray] = []
    for contraction_idx in range(shell.n_contractions):
        coeff_vector = shell.coefficients[contraction_idx]
        cart_functions: Dict[Tuple[int, int, int], np.ndarray] = {}
        for mon in monomials:
            primitives, overlap = primitive_cache[mon]
            cart_functions[mon] = contract_gaussians(primitives, overlap, coeff_vector)
        for combo in combos:
            orb = np.zeros_like(X)
            for weight, mon in combo:
                orb = orb + weight * cart_functions[mon]
            norm_sq = float(np.sum(np.abs(orb) ** 2) * dV)
            if norm_sq < EPS:
                functions.append(orb * 0.0)
            else:
                functions.append(orb / np.sqrt(norm_sq))
    return functions


def total_basis_functions(atoms: Sequence[AtomSpec]) -> int:
    count = 0
    for atom in atoms:
        for shell in atom.shells:
            count += shell.n_contractions * degeneracy(shell.angular_momentum, shell.is_pure)
    return count
