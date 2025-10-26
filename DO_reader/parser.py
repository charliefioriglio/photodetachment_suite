from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .basis import (
    ANGULAR_LETTER_TO_L,
    AtomSpec,
    DysonInfo,
    QChemData,
    ShellSpec,
    total_basis_functions,
)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
PURECART_PATTERN = re.compile(r"purecart\s*=\s*\"(?P<digits>\d+)\"")
BASIS_FN_PATTERN = re.compile(r"There are\s+(?P<n_shells>\d+) shells and (?P<n_basis>\d+) basis functions")
NORM_PATTERN = re.compile(
    r"(?P<side>Left|Right)\s*(?:Dyson)?\s*norm[^=]*=\s*(?P<value>[+\-0-9EeDd\.]+)"
)


@dataclass
class GeometryRecord:
    symbol: str
    coord_angstrom: np.ndarray

    @property
    def coord_bohr(self) -> np.ndarray:
        return self.coord_angstrom * ANGSTROM_TO_BOHR


def parse_float(token: str) -> float:
    token = token.replace("D", "E").replace("d", "E")
    return float(token)


def extract_geometry(lines: Sequence[str]) -> List[GeometryRecord]:
    geometry: List[GeometryRecord] = []
    for idx, line in enumerate(lines):
        if "Standard Nuclear Orientation" in line:
            geometry = []
            cursor = idx + 3
            while cursor < len(lines):
                row = lines[cursor].strip()
                cursor += 1
                if not row or row.startswith("-"):
                    continue
                parts = row.split()
                if len(parts) < 5 or not parts[0].isdigit():
                    break
                symbol = parts[1]
                coords = np.array([parse_float(x) for x in parts[2:5]], dtype=float)
                geometry.append(GeometryRecord(symbol=symbol, coord_angstrom=coords))
            # ensure we use geometry from final occurrence
    if not geometry:
        raise ValueError("Failed to locate molecular geometry in Q-Chem output")
    return geometry


def extract_basis_block(lines: Sequence[str]) -> List[str]:
    start = None
    for idx, line in enumerate(lines):
        if "Basis set in general basis input format" in line:
            start = idx
            break
    if start is None:
        raise ValueError("Could not find basis block in Q-Chem output")
    block: List[str] = []
    cursor = start
    while cursor < len(lines):
        line = lines[cursor]
        block.append(line.rstrip("\n"))
        if line.strip().startswith("$end"):
            break
        cursor += 1
    return block


def resolve_pure_map(purecart: str | None, overrides: Dict[int, bool] | None) -> Dict[int, bool]:
    mapping: Dict[int, bool] = {}
    if purecart:
        digits = [int(char) for char in purecart if char.isdigit()]
        l_values = [2, 3, 4, 5]
        for l, digit in zip(l_values, digits):
            mapping[l] = digit == 1
    else:
        # default to pure spherical for d and above
        for l in (2, 3, 4, 5):
            mapping[l] = True
    if overrides:
        mapping.update(overrides)
    return mapping


def parse_shell(
    shell_lines: List[str],
    symbol: str,
    center_bohr: np.ndarray,
    is_pure: bool,
    shell_index: int,
) -> ShellSpec:
    header = shell_lines[0].split()
    if len(header) < 3:
        raise ValueError(f"Malformed shell header: '{shell_lines[0]}'")
    ang_letter = header[0].upper()
    if ang_letter not in ANGULAR_LETTER_TO_L:
        raise ValueError(f"Unsupported angular momentum label '{ang_letter}'")
    angular_momentum = ANGULAR_LETTER_TO_L[ang_letter]
    n_prim = int(header[1])
    scale = parse_float(header[2])

    exponent_lines = shell_lines[1 : 1 + n_prim]
    exponents = np.array(
        [parse_float(row.split()[0]) for row in exponent_lines], dtype=float
    )
    coeff_rows: List[List[float]] = []
    for row in exponent_lines:
        pieces = row.split()[1:]
        coeff_rows.append([parse_float(x) for x in pieces])
    coeff_matrix = scale * np.array(coeff_rows, dtype=float).T

    label = f"{symbol} {ang_letter}{shell_index}"
    return ShellSpec(
        angular_momentum=angular_momentum,
        exponents=exponents,
        coefficients=coeff_matrix,
        is_pure=is_pure if angular_momentum >= 2 else True,
        symbol=symbol,
        label=label,
        center_bohr=center_bohr,
    )


def parse_basis(
    basis_lines: Sequence[str], geometry: Sequence[GeometryRecord], pure_map: Dict[int, bool]
) -> List[AtomSpec]:
    atoms: List[AtomSpec] = []
    geom_iter = iter(geometry)
    current_atom: AtomSpec | None = None
    shell_counter: Dict[str, int] = {}

    cursor = 0
    while cursor < len(basis_lines):
        line = basis_lines[cursor].strip()
        if (
            not line
            or line.startswith("-")
            or line.startswith("$")
            or line.startswith("Basis set")
            or line.startswith("Requested basis")
            or line.startswith("Compound shells")
        ):
            cursor += 1
            continue
        if line == "****":
            current_atom = None
            cursor += 1
            continue
        tokens = line.split()
        if len(tokens) == 2 and tokens[1].isdigit():
            symbol = tokens[0]
            try:
                geom = next(geom_iter)
            except StopIteration as exc:
                raise ValueError("Basis block defines more atoms than geometry") from exc
            if geom.symbol != symbol:
                raise ValueError(
                    f"Basis atom '{symbol}' does not match geometry atom '{geom.symbol}'"
                )
            current_atom = AtomSpec(
                symbol=symbol,
                index=len(atoms),
                center_bohr=geom.coord_bohr,
            )
            atoms.append(current_atom)
            shell_counter[symbol] = 0
            cursor += 1
            continue
        if current_atom is None:
            raise ValueError("Encountered shell definition before atom header in basis block")
        ang_letter = tokens[0].upper()
        l = ANGULAR_LETTER_TO_L.get(ang_letter)
        if l is None:
            raise ValueError(f"Unsupported shell type '{ang_letter}'")
        n_prim = int(tokens[1])
        block = [basis_lines[cursor]] + [basis_lines[cursor + i + 1] for i in range(n_prim)]
        cursor += n_prim + 1
        shell_counter[current_atom.symbol] += 1
        shell = parse_shell(
            block,
            symbol=current_atom.symbol,
            center_bohr=current_atom.center_bohr,
            is_pure=pure_map.get(l, True),
            shell_index=shell_counter[current_atom.symbol],
        )
        current_atom.shells.append(shell)
    return atoms


def parse_dyson_coefficients(lines: Sequence[str], n_basis: int) -> List[DysonInfo]:
    dyson_list: List[DysonInfo] = []
    idx = 0
    norm_pairs = extract_dyson_norms(lines)
    norm_iter = iter(norm_pairs)
    while idx < len(lines):
        line = lines[idx]
        if "Decomposition over AOs for the" not in line:
            idx += 1
            continue
        label = line.split("Decomposition over AOs for the", 1)[1].strip().strip(":")
        idx += 1
        coeffs: List[float] = []
        while idx < len(lines) and len(coeffs) < n_basis:
            row = lines[idx].strip()
            idx += 1
            if not row:
                continue
            if "Decomposition over AOs for the" in row:
                idx -= 1
                break
            parts = row.split()
            for part in parts:
                if len(coeffs) < n_basis:
                    coeffs.append(parse_float(part))
        if len(coeffs) != n_basis:
            raise ValueError(
                f"Expected {n_basis} Dyson coefficients for '{label}', found {len(coeffs)}"
            )
        dyson_list.append(DysonInfo(label=label, coefficients=np.array(coeffs)))
        try:
            left_norm, right_norm = next(norm_iter)
        except StopIteration:
            left_norm = right_norm = None
        dyson_list[-1].left_norm = left_norm
        dyson_list[-1].right_norm = right_norm
    if not dyson_list:
        raise ValueError("No Dyson orbital decompositions found in Q-Chem output")
    return dyson_list


def extract_dyson_norms(lines: Sequence[str]) -> List[tuple[float | None, float | None]]:
    norms: List[tuple[float | None, float | None]] = []
    current_left: float | None = None
    current_right: float | None = None
    for line in lines:
        match = NORM_PATTERN.search(line)
        if not match:
            continue
        value = parse_float(match.group("value"))
        side = match.group("side").lower()
        if side == "left":
            current_left = value
        else:
            current_right = value
        if current_left is not None and current_right is not None:
            norms.append((current_left, current_right))
            current_left = current_right = None
    if current_left is not None or current_right is not None:
        norms.append((current_left, current_right))
    return norms


def load_qchem_output(
    path: str | Path,
    pure_overrides: Dict[int, bool] | None = None,
) -> QChemData:
    text = Path(path).read_text()
    lines = text.splitlines()

    geometry = extract_geometry(lines)
    basis_lines = extract_basis_block(lines)

    purecart_match = PURECART_PATTERN.search(text)
    purecart_digits = purecart_match.group("digits") if purecart_match else None
    pure_map = resolve_pure_map(purecart_digits, pure_overrides)

    atoms = parse_basis(basis_lines, geometry, pure_map)

    basis_match = BASIS_FN_PATTERN.search(text)
    expected_n_basis = None
    if basis_match:
        expected_n_basis = int(basis_match.group("n_basis"))

    n_basis = total_basis_functions(atoms)
    if expected_n_basis is not None and expected_n_basis != n_basis:
        raise ValueError(
            f"Basis function count mismatch: parsed {n_basis}, expected {expected_n_basis}"
        )

    dyson = parse_dyson_coefficients(lines, n_basis)
    return QChemData(
        atoms=atoms,
        dyson_orbitals=dyson,
        n_basis_functions=n_basis,
        pure_map=pure_map,
    )
