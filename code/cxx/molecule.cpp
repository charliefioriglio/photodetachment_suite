#include "molecule.h"
#include <iostream>

void Molecule::add_atom(std::string symbol, int idx, double x, double y, double z) {
    AtomicCenter atom;
    atom.symbol = symbol;
    atom.index = idx;
    atom.x = x;
    atom.y = y;
    atom.z = z;
    atoms.push_back(atom);
}

void Molecule::add_shell_to_atom(int atom_idx, int l, bool pure, const std::vector<double>& exps, const std::vector<double>& coeffs) {
    if (atom_idx < 0 || static_cast<size_t>(atom_idx) >= atoms.size()) {
        std::cerr << "Error: Invalid atom index " << atom_idx << std::endl;
        return;
    }
    Shell s(l, pure, exps, coeffs);
    // Build happens in constructor
    atoms[atom_idx].shells.push_back(s);
}

void Molecule::shift_geometry(double dx, double dy, double dz) {
    for(auto& atom : atoms) {
        atom.x += dx;
        atom.y += dy;
        atom.z += dz;
    }
}

std::vector<Molecule::BasisFunctionRef> Molecule::flatten_basis() {
    std::vector<BasisFunctionRef> basis;
    for (const auto& atom : atoms) {
        for (const auto& shell : atom.shells) {
            for (const auto& func : shell.contracted_functions) {
                basis.push_back({&func, atom.x, atom.y, atom.z});
            }
        }
    }
    return basis;
}
