#ifndef MOLECULE_H
#define MOLECULE_H

#include <vector>
#include <string>
#include "shell.h"

struct AtomicCenter {
    std::string symbol;
    int index;
    double x, y, z; // Bohr
    std::vector<Shell> shells;
};

class Molecule {
public:
    std::vector<AtomicCenter> atoms;

    void add_atom(std::string symbol, int idx, double x, double y, double z);
    void add_shell_to_atom(int atom_idx, int l, bool pure, const std::vector<double>& exps, const std::vector<double>& coeffs);
    
    // Shift all atoms by (dx, dy, dz)
    void shift_geometry(double dx, double dy, double dz);

    // Flattened list of all basis functions in the molecule
    // This maps the global index mu to a specific function on a specific atom
    struct BasisFunctionRef {
        const ContractedGaussian* func;
        double cx, cy, cz;
    };

    std::vector<BasisFunctionRef> flatten_basis();
};

#endif // MOLECULE_H
