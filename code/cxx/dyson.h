#ifndef DYSON_H
#define DYSON_H

#include "molecule.h"
#include <string>
#include <vector>

class Dyson {
public:
    Molecule* mol;
    std::vector<double> coefficients; // Expansion coefficients (C_mu)
    std::string label;
    
    // Derived properties
    double normalization_factor = 1.0;
    double qchem_norm = 1.0; // Norm from Q-Chem output (for XS prefactor)
    
    Dyson(Molecule* molecule, const std::vector<double>& coeffs, std::string lbl = "DO");

    double evaluate(double x, double y, double z) const;
    
    // Normalization and Centroid
    // Requires a grid-like evaluation integration
    // We can do this using a bounding box numeric integration
    void renormalize(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, double step);
    
    struct Vector3 { double x, y, z; };
    Vector3 get_centroid(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, double step) const;
    
    // Update cached basis coordinates after moving molecule
    void update_geometry();
    
private:
    // Cached flattened basis for efficiency
    // Optimisation: Store only basis functions with non-zero coefficients
    std::vector<Molecule::BasisFunctionRef> active_basis_refs;
    std::vector<double> active_coefficients;
    
    // Original full lists (kept if needed, but we mostly use active ones)
    std::vector<Molecule::BasisFunctionRef> basis_refs; // Full list
};

#endif // DYSON_H
