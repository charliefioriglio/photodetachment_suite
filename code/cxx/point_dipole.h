#ifndef POINT_DIPOLE_H
#define POINT_DIPOLE_H

#include <vector>
#include <complex>
#include <map>
#include "dyson.h" // For vector types if needed, or just <vector>
#include "wigner_3j.h"

// Helper struct for caching eigensystems
struct Eigensystem {
    std::vector<double> eigvals;
    std::vector<std::vector<double>> eigvecs; // Columns are eigenvectors
    std::vector<int> l_vals;
};

class PointDipole {
private:
    double D; // Dipole moment in atomic units
    Wigner3J w3j;
    
    // Cache for eigensystems: key = (lam, l_max)
    // Note: D is fixed per instance, but lam and l_max vary.
    // Using a simple map with string key or tuple equivalent.
    std::map<std::pair<int, int>, Eigensystem> eigen_cache;
    
public:
    Eigensystem GetEigensystem(int lam, int l_max);

    PointDipole(double dipole_magnitude);

    // Evaluate the continuum wavefunction at a specific point for a specific k-vector
    // Sums over all allowed modes.
    // Matches python: build_directional_wavefunction
    std::complex<double> EvaluateDirectional(
        const double* k_vec,
        double energy_au,
        double x, double y, double z,
        int l_max = 10
    );

    // Evaluate the radial part of a specific eigenmode N
    // Returns: R_N(r)
    double EvaluateRadialMode(int lam, int N, double k, double r, int l_max=10);
};

#endif // POINT_DIPOLE_H
