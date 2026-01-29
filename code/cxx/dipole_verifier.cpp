#include "point_dipole.h"
#include "math_special.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to compute Omega_N(theta, phi=0) for a given mode
std::complex<double> ComputeOmega(const Eigensystem& sys, int N, int lam, double theta) {
    std::complex<double> val = 0.0;
    int n_basis = sys.l_vals.size();
    if (N >= n_basis) return 0.0;
    
    // Sum over l basis
    for (int i = 0; i < n_basis; ++i) {
        int l = sys.l_vals[i];
        double coeff = sys.eigvecs[i][N]; // N-th eigenvector, i-th component
        std::complex<double> Y = MathSpecial::SphericalHarmonicY(l, lam, theta, 0.0);
        val += coeff * Y;
    }
    return val;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    
    std::vector<double> D_vals = {0.0, 0.1, 0.25};
    int l_max = 5;
    double k = 1.0; 
    
    std::ofstream rad_file("dipole_radial_verification.csv");
    rad_file << "D,m,N,r,Re_Radial,Re_Ref_Bessel,L_eff,l_pure\n";

    std::ofstream ang_file("dipole_angular_verification.csv");
    ang_file << "D,m,N,theta,Re_Omega,Re_Ref_Ylm,L_eff,l_pure\n";
    
    // We will verify for m=0.
    int lam = 0;
    
    // Modes N=0, N=1 (Correspond to l=0, l=1 for D=0, m=0)
    std::vector<int> target_modes = {0, 1}; 
    
    for (double D_param : D_vals) {
        double dipole_strength = 2.0 * D_param; // Just for info, D_param is what goes into PointDipole
        // Wait, PointDipole takes "dipole_magnitude". The python code uses "D".
        // The Hamiltonian uses -2*D*...
        // The user says "D is the parameter inputted, and dipole strength = 2 * D".
        // Ensure PointDipole uses D as the parameter in the coupling term.
        // My PointDipole implementation uses D directly in coupling. Correct.
        
        PointDipole pd(D_param);
        Eigensystem sys = pd.GetEigensystem(lam, l_max);
        
        for (int N : target_modes) {
            if (N >= (int)sys.eigvals.size()) continue;
            
            double eigval = sys.eigvals[N];
            double L_eff = 0.5 * (-1.0 + std::sqrt(1.0 + 4.0 * eigval));
            int l_pure = std::abs(lam) + N; // nominal l
            
            // 1. Angular Scan (Theta 0 to Pi)
            int n_theta = 100;
            for (int i=0; i<=n_theta; ++i) {
                double theta = M_PI * i / n_theta;
                std::complex<double> omega = ComputeOmega(sys, N, lam, theta);
                std::complex<double> ref_Y = MathSpecial::SphericalHarmonicY(l_pure, lam, theta, 0.0);
                
                ang_file << D_param << "," << lam << "," << N << "," << theta << "," 
                         << std::real(omega) << "," << std::real(ref_Y) << "," 
                         << L_eff << "," << l_pure << "\n";
            }
            
            // 2. Radial Scan (r 0 to 20)
            int n_r = 200;
            for (int i=1; i<=n_r; ++i) {
                double r = 20.0 * i / n_r;
                double kr = k * r;
                
                // Point Dipole Radial: j_{L_eff}(kr) (scaled properly? usually just the bessel part for shape)
                // The wavefunction has prefactors, but user asks for "radial functions". 
                // Let's plot the raw Bessel part: sqrt(pi/2x) J_{L+0.5}
                double bessel_pd = 0.0;
                double J_nu = MathSpecial::CylBesselJ(L_eff + 0.5, kr);
                bessel_pd = std::sqrt(M_PI / (2.0 * kr)) * J_nu;
                
                // Reference Grid Radial: j_l(kr)
                double bessel_ref = MathSpecial::SphericalBesselJ(l_pure, kr);
                
                rad_file << D_param << "," << lam << "," << N << "," << r << "," 
                         << bessel_pd << "," << bessel_ref << "," 
                         << L_eff << "," << l_pure << "\n";
            }
        }
    }
    
    rad_file.close();
    ang_file.close();
    std::cout << "Verification data written to CSV files." << std::endl;
    return 0;
}
