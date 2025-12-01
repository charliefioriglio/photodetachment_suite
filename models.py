"""Common data structures for photodetachment calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class OrbitalPair:
    """Left/right Dyson orbitals defined on a common grid."""

    left: np.ndarray
    right: np.ndarray | None = None
    left_norm: float | None = None
    right_norm: float | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        left = np.asarray(self.left, dtype=np.complex128)
        object.__setattr__(self, "left", left)
        if self.right is not None:
            right = np.asarray(self.right, dtype=np.complex128)
            if right.shape != left.shape:
                raise ValueError("Left and right orbitals must share the same shape")
            object.__setattr__(self, "right", right)
        if self.left_norm is not None and self.left_norm < 0:
            raise ValueError("left_norm must be non-negative")
        if self.right_norm is not None and self.right_norm < 0:
            raise ValueError("right_norm must be non-negative")
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @property
    def right_or_left(self) -> np.ndarray:
        """Return the available orbital (right when present, otherwise left)."""

        return self.right if self.right is not None else self.left

    def scaled_left(self) -> np.ndarray:
        """Return the left orbital scaled by ``left_norm`` if provided."""

        factor = self.left_norm if self.left_norm is not None else 1.0
        return self.left * factor

    def scaled_right(self) -> np.ndarray:
        """Return the right orbital scaled by ``right_norm`` or fallback left."""

        if self.right is None:
            return self.scaled_left()
        factor = self.right_norm if self.right_norm is not None else 1.0
        return self.right * factor


@dataclass(frozen=True)
class TransitionChannel:
    """Photodetachment channel metadata."""

    label: str
    binding_energy_ev: float
    orbitals: Sequence[OrbitalPair] = field(default_factory=tuple)
    franck_condon: float = 1.0

    def __post_init__(self) -> None:
        if self.binding_energy_ev < 0.0:
            raise ValueError("binding_energy_ev must be non-negative")
        if self.franck_condon < 0.0:
            raise ValueError("franck_condon must be non-negative")
        if not self.orbitals:
            raise ValueError("At least one orbital pair is required")
        object.__setattr__(self, "orbitals", tuple(self.orbitals))

    @property
    def franck_condon_squared(self) -> float:
        """Return :math:`|FC|^2`, used repeatedly in cross sections."""

        return self.franck_condon ** 2


__all__ = ["OrbitalPair", "TransitionChannel"]
