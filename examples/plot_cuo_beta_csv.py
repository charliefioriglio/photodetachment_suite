"""Plot CuO beta-anisotropy results from saved CSV files."""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "cn_beta",
        help="Directory containing beta result CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination file to save the plot",
    )
    parser.add_argument(
        "--title",
        default="CuO photoelectron anisotropy",
        help="Plot title",
    )
    return parser.parse_args(argv)


def load_beta_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    e_ke = data[:, 0]
    beta = data[:, 2]
    return e_ke, beta


def extract_strength(path: Path) -> tuple[float, str]:
    name = path.stem
    if "plane_wave" in name:
        return (-1.0, "Plane wave")

    match = re.search(r"(\d+(?:[\._]\d+)?)", name)
    if match:
        raw_value = match.group(1).replace("_", ".")
        try:
            strength = float(raw_value) * 0.1
            return (strength, f"Point dipole {strength:.2f} au")
        except ValueError:
            pass

    return (float("inf"), name.replace("cuo_beta_", "").replace("_", " "))


def collect_beta_files(data_dir: Path) -> list[tuple[Path, str]]:
    csv_paths = sorted(data_dir.glob("cuo_beta_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    ordered: list[tuple[float, Path, str]] = []
    for path in csv_paths:
        strength, label = extract_strength(path)
        ordered.append((strength, path, label))

    ordered.sort(key=lambda item: item[0])
    return [(path, label) for _, path, label in ordered]


def plot_beta_curves(series: list[tuple[str, np.ndarray, np.ndarray]], title: str) -> plt.Figure:
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
    fig, ax = plt.subplots(figsize=(6.5, 4.25))

    for label, e_ke, beta in series:
        ax.plot(e_ke, beta, label=label, linewidth=2.0, color=next(colors))

    ax.set_xlabel("eKE (eV)")
    ax.set_ylabel(r"$\beta$")
    ax.set_ylim(-1.0, 2.0)
    ax.grid(False)
    ax.legend(frameon=False)
    ax.set_title(title)

    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    beta_files = collect_beta_files(args.data_dir)

    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for path, label in beta_files:
        e_ke, beta = load_beta_curve(path)
        series.append((label, e_ke, beta))

    fig = plot_beta_curves(series=series, title=args.title)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
