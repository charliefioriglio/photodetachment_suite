"""Visualize Dyson orbitals extracted with DO_reader."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from photodetachment_suite.DO_reader import DysonOrbitalBuilder, UniformGrid, load_qchem_output

ELEMENT_COLORS = {
    "H": "white",
    "C": "dimgray",
    "N": "royalblue",
    "O": "red"
}

EPS = 1.0e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qchem_output", help="Path to Q-Chem Dyson job output file")
    parser.add_argument(
        "--orbit",
        default="0",
        help="Dyson orbital selector (index or short label). Use --list to see options.",
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=12.0,
        help="Half-width of the cubic grid (in the chosen units)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=121,
        help="Number of grid points per axis (minimum 3)",
    )
    parser.add_argument(
        "--units",
        choices=["bohr", "angstrom"],
        default="bohr",
        help="Units of the visualization grid",
    )
    parser.add_argument(
        "--iso-fraction",
        type=float,
        default=0.02,
        help="Isosurface level as a fraction of the maximum |psi|",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the rendered figure",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figure (useful when saving to file)",
    )
    parser.add_argument(
        "--no-recenter",
        action="store_true",
        help="Skip density recentring iterations",
    )
    parser.add_argument("--max-iter", type=int, default=3, help="Maximum recenter iterations")
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-6,
        help="Centroid tolerance (Bohr) for recentring termination",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available Dyson orbitals in the Q-Chem output and exit",
    )
    parser.add_argument(
        "--title",
        help="Optional title for the plot. Defaults to file stem + orbital label.",
    )
    parser.add_argument(
        "--backend",
        help="Matplotlib backend override (e.g. Agg)",
    )
    return parser.parse_args(argv)


def ensure_backend(backend: str | None, no_show: bool) -> None:
    import matplotlib

    if backend:
        matplotlib.use(backend)
    elif no_show:
        matplotlib.use("Agg")


def make_grid(extent: float, points: int, units: str) -> UniformGrid:
    if points < 3:
        raise ValueError("Grid requires at least three points per axis")
    axis = np.linspace(-extent, extent, points)
    return UniformGrid(axis, axis, axis, unit=units)


def pick_selector(arg: str) -> str | int:
    return int(arg) if arg.isdigit() else arg


def element_color(symbol: str) -> str:
    base = ELEMENT_COLORS.get(symbol.capitalize())
    if base:
        return base

    # Generate a deterministic fallback color using matplotlib's qualitative map.
    import matplotlib.cm

    cmap = matplotlib.cm.get_cmap("tab20")
    idx = hash(symbol) % cmap.N
    rgba = cmap(idx)
    return tuple(float(channel) for channel in rgba[:3])


def plot_orbital(
    result,
    units: str,
    iso_fraction: float,
    title: str | None,
    show_atoms: bool = True,
):
    from matplotlib import pyplot as plt
    from skimage.measure import marching_cubes

    grid = result.grid if units == "bohr" else result.grid.to_unit("angstrom")
    psi = result.psi

    abs_max = float(np.max(np.abs(psi)))
    if abs_max < EPS:
        raise ValueError("Wavefunction amplitude is identically zero on the grid")

    iso_level = iso_fraction * abs_max
    if iso_level <= 0.0:
        raise ValueError("Isosurface fraction must be positive")

    dx, dy, dz = grid.spacing
    origin = np.array([grid.x.min(), grid.y.min(), grid.z.min()])

    verts_pos, faces_pos, _, _ = marching_cubes(psi, level=iso_level, spacing=(dx, dy, dz))
    verts_pos += origin

    verts_neg = faces_neg = None
    if np.min(psi) < 0.0:
        verts_neg, faces_neg, _, _ = marching_cubes(
            psi, level=-iso_level, spacing=(dx, dy, dz)
        )
        verts_neg += origin

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        verts_pos[:, 0],
        verts_pos[:, 1],
        verts_pos[:, 2],
        triangles=faces_pos,
        color="royalblue",
        alpha=0.65,
        linewidth=0.0,
    )

    if verts_neg is not None:
        ax.plot_trisurf(
            verts_neg[:, 0],
            verts_neg[:, 1],
            verts_neg[:, 2],
            triangles=faces_neg,
            color="indianred",
            alpha=0.55,
            linewidth=0.0,
        )

    if show_atoms:
        centers = (
            result.atom_centers_bohr
            if units == "bohr"
            else result.atom_centers_angstrom()
        )
        for symbol, center in zip(result.atom_symbols, centers):
            ax.scatter(*center, color=element_color(symbol), s=120, edgecolor="k", label=symbol)

    ax.set_xlim(grid.x.min(), grid.x.max())
    ax.set_ylim(grid.y.min(), grid.y.max())
    ax.set_zlim(grid.z.min(), grid.z.max())
    extents = (
        grid.x.max() - grid.x.min(),
        grid.y.max() - grid.y.min(),
        grid.z.max() - grid.z.min(),
    )
    ax.set_box_aspect(extents)

    label_unit = "Bohr" if units == "bohr" else "Å"
    ax.set_xlabel(f"X ({label_unit})")
    ax.set_ylabel(f"Y ({label_unit})")
    ax.set_zlabel(f"Z ({label_unit})")

    if title:
        ax.set_title(title)

    if show_atoms:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=9)

    return fig


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_backend(args.backend, args.no_show)

    data = load_qchem_output(args.qchem_output)
    builder = DysonOrbitalBuilder(data)

    if args.list:
        for idx, info in enumerate(builder.data.dyson_orbitals):
            print(f"{idx}: {info.short_label()}  ({info.display_label()})")
        return

    selector = pick_selector(args.orbit)
    grid = make_grid(args.extent, args.points, args.units)

    result = builder.build_orbital(
        selector=selector,
        grid=grid,
        recenter=not args.no_recenter,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    title = args.title
    if not title:
        title = f"{Path(args.qchem_output).stem} – {result.label}"

    fig = plot_orbital(result, units=args.units, iso_fraction=args.iso_fraction, title=title)

    print(f"Centroid (bohr): {result.centroid_bohr}")
    print(f"Centroid (Å): {result.centroid_angstrom()}")
    print(f"Normalization (pre-scaling): {result.normalization:.6g}")
    if result.left_norm is not None:
        print(f"Left norm (Q-Chem): {result.left_norm:.6g}")
    if result.right_norm is not None:
        print(f"Right norm (Q-Chem): {result.right_norm:.6g}")
    print(f"Recentering iterations: {result.iterations}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {args.output}")

    from matplotlib import pyplot as plt

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
