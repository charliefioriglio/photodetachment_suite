#include "point_dipole.h"
#include "continuum.h"
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    std::cout << "Testing Point Dipole Continuum Implementation..." << std::endl;

    double x = 1.0;
    double y = 0.5;
    double z = -0.2;
    double r[3] = {x, y, z};
    
    double k_vec[3] = {0.0, 0.0, 1.0}; // k along z
    double E_au = 0.5; // E = k^2/2 -> k=1
    
    // D=0 Case: Should match Plane Wave (Expansion)
    std::cout << "Test 1: D=0 vs Plane Wave Expansion" << std::endl;
    PointDipole pd_zero(0.0);
    
    int l_max = 10;
    
    std::complex<double> val_pd = pd_zero.EvaluateDirectional(k_vec, E_au, x, y, z, l_max);
    std::complex<double> val_pwe = Continuum::EvaluatePlaneWaveExpansion(k_vec, r, l_max);
    
    std::cout << "Point Dipole (D=0): " << val_pd << std::endl;
    std::cout << "Plane Wave Exp:     " << val_pwe << std::endl;
    
    double diff = std::abs(val_pd - val_pwe);
    std::cout << "Difference: " << diff << std::endl;
    
    if (diff < 1e-4) {
        std::cout << "[PASS] D=0 matches PWE." << std::endl;
    } else {
        std::cout << "[FAIL] D=0 mismatch." << std::endl;
        return 1;
    }

    // D=2.5 Case: Just check it runs and differs
    std::cout << "\nTest 2: D=2.5 (Non-zero Dipole)" << std::endl;
    PointDipole pd_nonzero(2.5);
    std::complex<double> val_pd_nz = pd_nonzero.EvaluateDirectional(k_vec, E_au, x, y, z, l_max);
    std::cout << "Point Dipole (D=2.5): " << val_pd_nz << std::endl;
    
    if (std::abs(val_pd_nz - val_pd) > 1e-4) {
        std::cout << "[PASS] Non-zero D differs from D=0." << std::endl;
    } else {
        std::cout << "[FAIL] Non-zero D identical to D=0 (Implies coupling failed)." << std::endl;
        return 1;
    }

    return 0;
}
