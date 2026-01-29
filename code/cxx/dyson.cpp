#include "dyson.h"
#include <iostream>

Dyson::Dyson(Molecule* molecule, const std::vector<double>& coeffs, std::string lbl)
    : mol(molecule), coefficients(coeffs), label(lbl) {
    basis_refs = mol->flatten_basis();
    if (basis_refs.size() != coefficients.size()) {
        std::cerr << "Warning: Number of basis functions (" << basis_refs.size() 
                  << ") does not match coefficients (" << coefficients.size() << ")" << std::endl;
    }
    
    // Pre-filter significant coefficients
    active_basis_refs.reserve(basis_refs.size());
    active_coefficients.reserve(basis_refs.size());
    
    size_t limit = std::min(basis_refs.size(), coefficients.size());
    for(size_t i=0; i<limit; ++i) {
        if(std::abs(coefficients[i]) > 1.0e-12) { // 1e-12 threshold as per implementation plan
            active_basis_refs.push_back(basis_refs[i]);
            active_coefficients.push_back(coefficients[i]);
        }
    }
}

double Dyson::evaluate(double x, double y, double z) const {
    double val = 0.0;
    // Iterate only over active basis functions
    for (size_t i = 0; i < active_coefficients.size(); ++i) {
        val += active_coefficients[i] * active_basis_refs[i].func->evaluate(x, y, z, active_basis_refs[i].cx, active_basis_refs[i].cy, active_basis_refs[i].cz);
    }
    return val * normalization_factor;
}

void Dyson::renormalize(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, double step) {
    double norm_sq = 0.0;
    double dV = step * step * step;
    
    for (double x = xmin; x <= xmax; x += step) {
        for (double y = ymin; y <= ymax; y += step) {
            for (double z = zmin; z <= zmax; z += step) {
                double v = evaluate(x, y, z); // Uses current normalization_factor (1.0 initially)
                norm_sq += v * v * dV;
            }
        }
    }
    
    if (norm_sq > 1.0e-14) {
        normalization_factor = 1.0 / std::sqrt(norm_sq);
    } else {
        normalization_factor = 0.0;
    }
    std::cout << "Computed Norm: " << std::sqrt(norm_sq) << ". Renormalization factor: " << normalization_factor << std::endl;
}

Dyson::Vector3 Dyson::get_centroid(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, double step) const {
    double norm_sq = 0.0;
    double X = 0.0, Y = 0.0, Z = 0.0;
    double dV = step * step * step;

    for (double x = xmin; x <= xmax; x += step) {
        for (double y = ymin; y <= ymax; y += step) {
            for (double z = zmin; z <= zmax; z += step) {
                double val = evaluate(x, y, z);
                double p2 = val * val;
                norm_sq += p2 * dV;
                X += x * p2 * dV;
                Y += y * p2 * dV;
                Z += z * p2 * dV;
            }
        }
    }
    
    if (norm_sq < 1.0e-14) return {0,0,0};
    return {X/norm_sq, Y/norm_sq, Z/norm_sq};
}

void Dyson::update_geometry() {
    // Re-flatten to get new coordinates
    basis_refs = mol->flatten_basis();
    
    // Re-filter active refs (logic duplicated from constructor, but safe)
    active_basis_refs.clear();
    active_coefficients.clear();
    
    active_basis_refs.reserve(basis_refs.size());
    active_coefficients.reserve(basis_refs.size());
    
    size_t limit = std::min(basis_refs.size(), coefficients.size());
    for(size_t i=0; i<limit; ++i) {
        if(std::abs(coefficients[i]) > 1.0e-12) {
            active_basis_refs.push_back(basis_refs[i]);
            active_coefficients.push_back(coefficients[i]);
        }
    }
}
