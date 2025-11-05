from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from .dyson import DysonOrbitalBuilder, UniformGrid
from .parser import load_qchem_output

DESCRIPTION = "Build normalized, recentered Dyson orbitals on user-defined grids."


def parse_axis(arg: Tuple[str, str, str]) -> np.ndarray:
    xmin, xmax, npts = arg
    start = float(xmin)
    stop = float(xmax)
    count = int(npts)
    if count < 2:
        raise argparse.ArgumentTypeError("Grid axis must contain at least two points")
    return np.linspace(start, stop, count)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("qchem_output", help="Path to Q-Chem Dyson job output file")
    parser.add_argument("output", help="Output .npz file to store orbital data")
    parser.add_argument(
        "--orbit",
        help="Dyson orbital selector (short label or index)",
        default="0",
    )
    parser.add_argument("--x", nargs=3, metavar=("XMIN", "XMAX", "NX"), required=True)
    parser.add_argument("--y", nargs=3, metavar=("YMIN", "YMAX", "NY"), required=True)
    parser.add_argument("--z", nargs=3, metavar=("ZMIN", "ZMAX", "NZ"), required=True)
    parser.add_argument(
        "--units",
        choices=["bohr", "angstrom"],
        default="bohr",
        help="Units for grid coordinates",
    )
    parser.add_argument(
        "--no-recenter",
        action="store_true",
        help="Skip recentring the orbital density",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Maximum number of recenter iterations",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-6,
        help="Centroid tolerance in Bohr",
    )
    parser.add_argument(
        "--purecart",
        help="Override pure/cart settings as JSON, e.g. '{\"2\": true, \"3\": true}'",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available Dyson orbitals and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)

    overrides = None
    if args.purecart:
        overrides = {int(k): bool(v) for k, v in json.loads(args.purecart).items()}

    data = load_qchem_output(args.qchem_output, pure_overrides=overrides)
    builder = DysonOrbitalBuilder(data)

    if args.list:
        for idx, info in enumerate(builder.data.dyson_orbitals):
            print(f"{idx}: {info.short_label()}  ({info.display_label()})")
        return

    axis_x = parse_axis(tuple(args.x))
    axis_y = parse_axis(tuple(args.y))
    axis_z = parse_axis(tuple(args.z))

    grid = UniformGrid(axis_x, axis_y, axis_z, unit=args.units)

    selector: str | int
    if args.orbit.isdigit():
        selector = int(args.orbit)
    else:
        selector = args.orbit

    result = builder.build_orbital(
        selector=selector,
        grid=grid,
        recenter=not args.no_recenter,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        psi=result.psi,
        x=grid.x,
        y=grid.y,
        z=grid.z,
        centroid_bohr=result.centroid_bohr,
        centroid_angstrom=result.centroid_angstrom(),
        normalization=result.normalization,
        iterations=result.iterations,
        label=result.label,
        coefficients=result.coefficients,
        transition=result.transition if result.transition is not None else "",
        symmetry=result.symmetry if result.symmetry is not None else "",
        state_index=result.state_index if result.state_index is not None else "",
        side=result.side if result.side is not None else "",
    )

    print(f"Saved Dyson orbital '{result.label}' to {output_path}")
    print(f"Centroid (bohr): {result.centroid_bohr}")
    print(f"Centroid (angstrom): {result.centroid_angstrom()}")
    print(f"Normalization factor applied: {result.normalization}")
    if result.iterations:
        print(f"Recentering iterations: {result.iterations}")


if __name__ == "__main__":
    main()
