#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>

#include "physical_dipole.h"
#include "point_dipole.h"
#include "math_special.h"

// Usage: ./verify_radial D a E m mode_idx r_min r_max step output_file
// mode_idx: 0 means lowest L for that m.
int main(int argc, char** argv) {
    if (argc < 10) {
        std::cerr << "Usage: ./verify_radial D a E m mode_idx r_min r_max step output.csv" << std::endl;
        return 1;
    }
    
    double D = std::stod(argv[1]);
    double a = std::stod(argv[2]);
    double E = std::stod(argv[3]);
    int m = std::stoi(argv[4]);
    int mode_idx = std::stoi(argv[5]);
    double r_min = std::stod(argv[6]);
    double r_max = std::stod(argv[7]);
    double step = std::stod(argv[8]);
    std::string out_file = argv[9];
    
    double k = std::sqrt(2.0 * E);
    double c = k * a;
    
    // 1. Solve Physical Dipole System
    // Need l_max sufficient for convergence
    int l_max = 30; 
    
    PhysicalDipole phys(a, D);
    auto sol = phys.Solve(E, m, l_max);
    
    int n_modes = sol.ang_eigenvalues.size();
    if (mode_idx >= n_modes) {
        std::cerr << "Error: mode_idx " << mode_idx << " out of range (" << n_modes << ")" << std::endl;
        return 1;
    }
    
    const auto& rad_sol = sol.radial_solutions[mode_idx];
    
    // 2. Solve Point Dipole System (Reference)
    // Point Dipole Parameter D_point.
    // Hamiltonian term: -2 D / r^2. 
    // PointDipole class takes "dipole_magnitude". 
    // In PointDipole::H: coupling = 2.0 * D * ...
    // So pass D directly.
    PointDipole pd(D);
    Eigensystem sys_pd = pd.GetEigensystem(m, l_max);
    // Find mode with similar L character?
    // Point Dipole modes are mixtures too (for D!=0).
    // For D=0, they are pure L.
    // Index 0 of m-block is lowest L (L=m).
    
    double L_eff_pd = 0.0;
    bool pd_valid = false;
    if (mode_idx < (int)sys_pd.eigvals.size()) {
        double lam_pd = sys_pd.eigvals[mode_idx];
        // lam = l(l+1) for D=0.
        // L_eff = -0.5 + sqrt(0.25 + lam)
        if (lam_pd >= -0.25) {
             L_eff_pd = -0.5 + std::sqrt(0.25 + lam_pd);
             pd_valid = true;
        }
    }
    
    // 3. PWE Reference (L = pure)
    // Corresponds to l = m + mode_idx
    int l_pwe = std::abs(m) + mode_idx;
    
    std::ofstream out(out_file);
    out << "r,S_phys_real,S_phys_imag,S_point,S_pwe\n";
    
    for (double r = r_min; r <= r_max; r += step) {
        if (r < 1e-6) continue;
        
        // Physical Dipole
        // xi = r/a. (Actually only valid r >= a -> xi >= 1).
        // If r < a, Physical Dipole undefined in this coord system unless inside?
        // Prolate coordinates: xi >= 1 covers space outside the line segment.
        // We will output 0 for r < a or clamp xi=1?
        // User wants to see "near field".
        // If a=1, r goes to 0? No, r=xi*a. Min r=a.
        // We cannot plot r < a using S(xi).
        // For r < a, we are "inside" the dipole separation?
        // Spheroidal coords cover all space. xi=1 is the line between foci.
        // So r_min should be a.
        
        std::complex<double> S_phys = 0.0;
        if (r >= a) {
            double xi = r / a;
            S_phys = PhysicalDipoleRadial::Evaluate(xi, rad_sol.coeffs, rad_sol.nu, c, m);
        }
        
        // Point Dipole: j_{L_eff}(kr)
        double val_pd = 0.0;
        if (pd_valid) {
            double kr = k * r;
            // sqrt(pi/2kr) J_{L+0.5}(kr)
            val_pd = std::sqrt(M_PI/(2.0*kr)) * MathSpecial::CylBesselJ(L_eff_pd+0.5, kr);
        }
        
        // PWE: j_l(kr)
        double val_pwe = MathSpecial::SphericalBesselJ(l_pwe, k * r);
        
        out << r << "," << S_phys.real() << "," << S_phys.imag() << "," << val_pd << "," << val_pwe << "\n";
    }
    
    out.close();
    return 0;
}
