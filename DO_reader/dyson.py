from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .basis import QChemData, enumerate_shell_functions
from ..models import OrbitalPair

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
EPS = 1.0e-12


@dataclass
class UniformGrid:
    x: np.ndarray  # 1D axis in Bohr
    y: np.ndarray
    z: np.ndarray
    unit: str = "bohr"

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        if self.unit not in {"bohr", "angstrom"}:
            raise ValueError("Grid unit must be 'bohr' or 'angstrom'")

    @property
    def spacing(self) -> tuple[float, float, float]:
        if len(self.x) < 2 or len(self.y) < 2 or len(self.z) < 2:
            raise ValueError("Each grid axis must contain at least two points")
        dx = float(self.x[1] - self.x[0])
        dy = float(self.y[1] - self.y[0])
        dz = float(self.z[1] - self.z[0])
        return dx, dy, dz

    def differential_volume(self) -> float:
        dx, dy, dz = self.spacing
        return dx * dy * dz

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, self.z, indexing="ij")

    def in_bohr(self) -> "UniformGrid":
        return self.to_unit("bohr")

    def shift(self, offset: Sequence[float]) -> "UniformGrid":
        offset = np.asarray(offset, dtype=float)
        return UniformGrid(self.x - offset[0], self.y - offset[1], self.z - offset[2], unit=self.unit)

    def to_unit(self, target: str) -> "UniformGrid":
        target = target.lower()
        if target not in {"bohr", "angstrom"}:
            raise ValueError("Target unit must be 'bohr' or 'angstrom'")
        if target == self.unit:
            return UniformGrid(self.x.copy(), self.y.copy(), self.z.copy(), unit=self.unit)
        if self.unit == "bohr" and target == "angstrom":
            factor = BOHR_TO_ANGSTROM
        elif self.unit == "angstrom" and target == "bohr":
            factor = ANGSTROM_TO_BOHR
        else:
            raise ValueError(f"Unsupported unit conversion from {self.unit} to {target}")
        return UniformGrid(self.x * factor, self.y * factor, self.z * factor, unit=target)


@dataclass
class EvaluationResult:
    psi: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    dV: float
    norm: float


@dataclass
class DysonBuildResult:
    label: str
    psi: np.ndarray
    grid: UniformGrid
    centroid_bohr: np.ndarray
    normalization: float
    iterations: int
    coefficients: np.ndarray
    atom_symbols: List[str]
    atom_centers_bohr: np.ndarray
    left_norm: float | None
    right_norm: float | None
    transition: str | None
    state_index: str | None
    symmetry: str | None
    side: str | None

    def to_orbital_pair(self, right: "DysonBuildResult | None" = None) -> OrbitalPair:
        """Convert the builder result into an :class:`OrbitalPair`."""

        left_scale = (self.left_norm if self.left_norm is not None else 1.0) * self.normalization

        if right is None:
            right_wave = self.psi
            right_scale = (
                (self.right_norm if self.right_norm is not None else 1.0)
                * self.normalization
            )
        else:
            right_wave = right.psi
            right_scale = (
                (right.right_norm if right.right_norm is not None else 1.0)
                * right.normalization
            )

        metadata: dict[str, np.ndarray | float | list[str]] = {
            "atom_symbols": list(self.atom_symbols),
            "atom_centers_bohr": self.atom_centers_bohr.copy(),
        }

        if self.atom_centers_bohr.size:
            midpoint = np.mean(self.atom_centers_bohr, axis=0)
            metadata["bond_midpoint_bohr"] = midpoint
            if self.atom_centers_bohr.shape[0] == 2:
                vec = self.atom_centers_bohr[0] - self.atom_centers_bohr[1]
                length = float(np.linalg.norm(vec))
                if length > EPS:
                    metadata["bond_vector_bohr"] = vec
                    metadata["bond_axis"] = vec / length
                    metadata["bond_length_bohr"] = length
                    metadata["half_bond_length_bohr"] = 0.5 * length

        return OrbitalPair(
            left=self.psi,
            right=right_wave,
            left_norm=left_scale,
            right_norm=right_scale,
            metadata=metadata,
        )

    def centroid_angstrom(self) -> np.ndarray:
        return self.centroid_bohr * BOHR_TO_ANGSTROM

    def atom_centers_angstrom(self) -> np.ndarray:
        return self.atom_centers_bohr * BOHR_TO_ANGSTROM


class DysonOrbitalBuilder:
    def __init__(self, data: QChemData):
        self.data = data
        self._original_centers = np.array([atom.center_bohr for atom in data.atoms], dtype=float)

    def available_orbitals(self) -> List[str]:
        return [info.short_label() for info in self.data.dyson_orbitals]

    def build_orbital(
        self,
        selector: str | int,
        grid: UniformGrid,
        recenter: bool = True,
        max_iter: int = 3,
        tol: float = 1.0e-6,
    ) -> DysonBuildResult:
        dyson = self.data.get_dyson(selector)
        grid_bohr = grid.in_bohr()
        centers = self._original_centers.copy()

        result = self._evaluate(dyson.coefficients, centers, grid_bohr)
        centroid = self._centroid(result.psi, result.X, result.Y, result.Z, result.dV)
        iterations = 0

        while recenter and np.linalg.norm(centroid) > tol and iterations < max_iter:
            grid_bohr = grid_bohr.shift(centroid)
            centers = centers - centroid
            result = self._evaluate(dyson.coefficients, centers, grid_bohr)
            centroid = self._centroid(result.psi, result.X, result.Y, result.Z, result.dV)
            iterations += 1

        return DysonBuildResult(
            label=dyson.short_label(),
            psi=result.psi,
            grid=grid_bohr,
            centroid_bohr=centroid,
            normalization=result.norm,
            iterations=iterations,
            coefficients=dyson.coefficients,
            atom_symbols=[atom.symbol for atom in self.data.atoms],
            atom_centers_bohr=centers,
            left_norm=dyson.left_norm,
            right_norm=dyson.right_norm,
            transition=dyson.transition,
            state_index=dyson.state_index,
            symmetry=dyson.symmetry,
            side=dyson.side,
        )

    def _evaluate(
        self,
        coefficients: np.ndarray,
        centers: np.ndarray,
        grid: UniformGrid,
    ) -> EvaluationResult:
        X, Y, Z = grid.mesh()
        dV = grid.differential_volume()
        ao_functions = self._assemble_ao_functions(centers, X, Y, Z, dV)
        if len(ao_functions) != coefficients.size:
            raise ValueError(
                f"Coefficient vector length {coefficients.size} does not match AO count {len(ao_functions)}"
            )
        psi = np.zeros_like(X)
        for coeff, ao in zip(coefficients, ao_functions):
            psi = psi + coeff * ao
        norm_sq = float(np.sum(np.abs(psi) ** 2) * dV)
        norm = np.sqrt(max(norm_sq, EPS))
        psi_normalized = psi / norm if norm > 0 else psi
        return EvaluationResult(psi=psi_normalized, X=X, Y=Y, Z=Z, dV=dV, norm=norm)

    def _assemble_ao_functions(
        self,
        centers: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        dV: float,
    ) -> List[np.ndarray]:
        ao_values: List[np.ndarray] = []
        for atom, center in zip(self.data.atoms, centers):
            for shell in atom.shells:
                ao_values.extend(
                    enumerate_shell_functions(shell, center, X, Y, Z, dV)
                )
        return ao_values

    @staticmethod
    def _centroid(
        psi: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        dV: float,
    ) -> np.ndarray:
        density = np.abs(psi) ** 2
        weight = float(np.sum(density) * dV)
        if weight < EPS:
            return np.zeros(3)
        cx = float(np.sum(density * X) * dV)
        cy = float(np.sum(density * Y) * dV)
        cz = float(np.sum(density * Z) * dV)
        return np.array([cx, cy, cz]) / weight
