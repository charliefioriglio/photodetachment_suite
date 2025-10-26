"""Plotting helpers for photodetachment observables."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .beta_calculator import BetaResult
from .cross_sections import CrossSectionResult


def _ensure_axes(ax):
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return None, ax


def plot_beta(result: BetaResult, *, ax=None, label: str | None = None):
    """Plot beta vs photon energy."""

    fig, axis = _ensure_axes(ax)
    axis.plot(result.energies_ev, result.beta, marker="o", label=label or r"$\beta$")
    axis.set_xlabel("Photon energy (eV)")
    axis.set_ylabel(r"$\beta$")
    axis.set_title("Anisotropy parameter vs photon energy")
    axis.grid(True, alpha=0.3)
    if label:
        axis.legend()
    return axis if fig is None else (fig, axis)


def plot_total_cross_section(
    result: CrossSectionResult,
    *,
    ax=None,
    include_channels: bool = False,
    channel_labels: Sequence[str] | None = None,
) -> tuple:
    """Plot total cross section (and optionally per-channel curves)."""

    fig, axis = _ensure_axes(ax)
    axis.plot(
        result.photon_energies_ev,
        result.total_cross_section,
        marker="o",
        label="Total" if include_channels else None,
    )
    axis.set_xlabel("Photon energy (eV)")
    axis.set_ylabel(r"$\sigma$ (a.u.)")
    axis.set_title("Photodetachment cross section")
    axis.grid(True, alpha=0.3)

    if include_channels:
        labels = channel_labels or result.channel_labels
        for idx, label in enumerate(labels):
            axis.plot(
                result.photon_energies_ev,
                result.per_channel[:, idx],
                marker="o",
                linestyle="--",
                label=label,
            )
        axis.legend()

    return axis if fig is None else (fig, axis)


def plot_relative_cross_sections(
    result: CrossSectionResult,
    *,
    ax=None,
    channel_labels: Sequence[str] | None = None,
) -> tuple:
    """Plot relative channel weights vs photon energy."""

    fig, axis = _ensure_axes(ax)
    fractions = result.relative()
    labels = channel_labels or result.channel_labels

    for idx, label in enumerate(labels):
        axis.plot(
            result.photon_energies_ev,
            fractions[:, idx],
            marker="o",
            label=label,
        )

    axis.set_xlabel("Photon energy (eV)")
    axis.set_ylabel("Relative weight")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Relative photodetachment cross sections")
    axis.grid(True, alpha=0.3)
    axis.legend()

    return axis if fig is None else (fig, axis)


__all__ = [
    "plot_beta",
    "plot_total_cross_section",
    "plot_relative_cross_sections",
]
