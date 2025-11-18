"""Utilities for computing photodetachment observables."""

from . import averaging_angle_grids
from .beta_calculator import BetaResult, calculate_beta
from .continuum import ContinuumType, get_continuum_function
from .cross_sections import (
    CrossSectionResult,
    RelativeCrossSectionResult,
    calculate_relative_cross_sections,
    calculate_total_cross_sections,
    calculate_relative_cross_sections_from_total,
)
from .grid import CartesianGrid
from .integration import integrate_scalar_field
from .models import OrbitalPair, TransitionChannel
from .physical_dipole import (
    PhysicalDipoleParameters,
    calculate_physical_dipole_cross_sections,
)
from .plots import (
    plot_beta,
    plot_relative_cross_sections,
    plot_total_cross_section,
)
from .rotation import rotation_matrix

__all__ = [
    "averaging_angle_grids",
    "BetaResult",
    "CrossSectionResult",
    "RelativeCrossSectionResult",
    "OrbitalPair",
    "TransitionChannel",
    "calculate_beta",
    "calculate_relative_cross_sections",
    "calculate_relative_cross_sections_from_total",
    "calculate_total_cross_sections",
    "calculate_physical_dipole_cross_sections",
    "ContinuumType",
    "get_continuum_function",
    "CartesianGrid",
    "integrate_scalar_field",
    "plot_beta",
    "plot_relative_cross_sections",
    "plot_total_cross_section",
    "rotation_matrix",
    "PhysicalDipoleParameters",
]
