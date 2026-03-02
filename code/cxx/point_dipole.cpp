#include "point_dipole.h"
#include "linalg.h"
#include "math_special.h"
#include <cmath>
#include <iostream>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

PointDipole::PointDipole(double dipole_magnitude) : D(dipole_magnitude) {}

Eigensystem PointDipole::GetEigensystem(int lam, int l_max) {
    if (eigen_cache.find({lam, l_max}) != eigen_cache.end()) {
        return eigen_cache[{lam, l_max}];
    }
    
    int l_min = std::abs(lam);
    std::vector<int> l_vals;
    for (int l = l_min; l <= l_max; ++l) l_vals.push_back(l);
    
    int n = l_vals.size();
    if (n == 0) return Eigensystem{};

    // Build Hamiltonian H
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    
    // 1. Diagonal: l(l+1)
    for (int i = 0; i < n; ++i) {
        H[i][i] = (double)(l_vals[i] * (l_vals[i] + 1));
    }
    
    // 2. Off-diagonal: Dipole coupling
    // Matches python: build_dipole_angular_matrix
    // coeff = -2 * D * sqrt((2l+1)(2l'+1)) * w3j(l', 1, l, 0, 0, 0) * w3j(l', 1, l, -lam, 0, lam)
    
    for (int i = 0; i < n; ++i) {
        int l = l_vals[i];
        
        // Check neighbors l' = l +/- 1
        int neighbors[] = {l - 1, l + 1};
        for (int lp : neighbors) {
            if (lp < l_min || lp > l_max) continue;
            
            // Find index j for lp
            int j = -1;
            for(int k=0; k<n; ++k) if(l_vals[k] == lp) { j = k; break; }
            if (j == -1) continue;
            
            double tj1 = w3j.get_3j_dipole(lp, l, 0, 0, 0);       // (lp 1 l // 0 0 0)
            double tj2 = w3j.get_3j_dipole(lp, l, -lam, 0, lam);  // (lp 1 l // -lam 0 lam)
            
            double coupling = -2.0 * D * std::sqrt((2.0 * l + 1.0) * (2.0 * lp + 1.0)) * tj1 * tj2;
            
            H[i][j] += coupling; 
        }
    }
    
    for (int i = 0; i < n; ++i) {
        int l = l_vals[i];
        int neighbors[] = {l - 1, l + 1};
        for (int lp : neighbors) {
            if (lp < l_min || lp > l_max) continue;
             int j = -1;
            for(int k=0; k<n; ++k) if(l_vals[k] == lp) { j = k; break; }
            if (j <= i) continue; // Only Upper Triangular
            
            double tj1 = w3j.get_3j_dipole(lp, l, 0, 0, 0);       
            double tj2 = w3j.get_3j_dipole(lp, l, -lam, 0, lam);  
            
            double coupling = 2.0 * D * std::sqrt((2.0 * l + 1.0) * (2.0 * lp + 1.0)) * tj1 * tj2;
            H[i][j] = coupling;
            H[j][i] = coupling;
        }
    }
    
    std::vector<double> eigvals;
    std::vector<std::vector<double>> eigvecs;
    LinearAlgebra::Jacobi(n, H, eigvals, eigvecs);
    
    Eigensystem res;
    res.eigvals = eigvals;
    res.eigvecs = eigvecs;
    res.l_vals = l_vals;
    
    eigen_cache[{lam, l_max}] = res;
    
    return res;
}

std::complex<double> PointDipole::EvaluateDirectional(
    const double* k_vec,
    double energy_au,
    double x, double y, double z,
    int l_max
) {
    if (energy_au <= 0.0) return 0.0;
    
    double r = std::sqrt(x*x + y*y + z*z);
    if (r < 1e-12) return 0.0;
    
    double theta = std::acos(z / r);
    double phi = std::atan2(y, x);
    
    double k_mag = std::sqrt(k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2]);
    if (k_mag < 1e-12) return 0.0; // Should be sqrt(2E)
    
    // Normalize k
    double k_hat[3] = {k_vec[0]/k_mag, k_vec[1]/k_mag, k_vec[2]/k_mag};
    double theta_k = std::acos(k_hat[2]);
    double phi_k = std::atan2(k_hat[1], k_hat[0]);
    
    std::complex<double> total_psi(0.0, 0.0);
    
    // Sum over lam (m)
    for (int lam = -l_max; lam <= l_max; ++lam) {
        Eigensystem sys = GetEigensystem(lam, l_max);
        if (sys.eigvals.empty()) continue;
        
        // Pre-compute harmonics for direction k
        int n_basis = sys.l_vals.size();
        std::vector<std::complex<double>> Y_lm_k(n_basis);
        for(int i=0; i<n_basis; ++i) {
            Y_lm_k[i] = MathSpecial::SphericalHarmonicY(sys.l_vals[i], lam, theta_k, phi_k);
        }
        
        // Loop over Eigenmodes N
        for (int N = 0; N < n_basis; ++N) {
            double eigval = sys.eigvals[N];
            
            // Filter modes
            if (eigval < -0.25) continue; // Bound state or unphysical
            
            std::complex<double> omega_dir = 0.0;
            for(int i=0; i<n_basis; ++i) {
                // sys.eigvecs[i][N] is the coefficient of l-basis i in mode N
                // Matrix columns are eigenvectors
                omega_dir += sys.eigvecs[i][N] * Y_lm_k[i];
            }
            
            if (std::abs(omega_dir) < 1e-12) continue;
            
            // Effective L
            double L_eff = 0.5 * (-1.0 + std::sqrt(1.0 + 4.0 * eigval));
            
            // Radial Part: j_Leff(kr)
            // Using std::cyl_bessel_j(nu, x) ?
            // Spherical Bessel j_n(x) = sqrt(pi/2x) J_{n+0.5}(x)
            // generalized for non-integer n -> L_eff
            // radial = sqrt(pi / (2*kr)) * J_{L_eff + 0.5}(kr)
            double kr = k_mag * r;
            // Avoid division by zero
            double radial = 0.0;
            if (kr > 1e-10) {
                 double bessel = MathSpecial::CylBesselJ(L_eff + 0.5, kr);
                 radial = std::sqrt(M_PI / (2.0 * kr)) * bessel;
            } else {
                 if (L_eff < 0.1) radial = 1.0; // Approx for s-wave? Limit x->0 j_0(x)=1
                 else radial = 0.0;
            }
            
            // Phase factor i^L_eff = exp(i * pi/2 * L_eff)
            std::complex<double> phase = std::exp(std::complex<double>(0.0, M_PI * 0.5 * L_eff));
            
            std::complex<double> weight = 4.0 * M_PI * phase * std::conj(omega_dir);
            
            // Mode spatial angular function Omega_N(r)
            std::complex<double> omega_r = 0.0;
            for(int i=0; i<n_basis; ++i) {
                std::complex<double> Y_lm_r = MathSpecial::SphericalHarmonicY(sys.l_vals[i], lam, theta, phi);
                omega_r += sys.eigvecs[i][N] * Y_lm_r;
            }
            
            total_psi += weight * radial * omega_r;
        }
    }
    
    return total_psi;
}

double PointDipole::EvaluateRadialMode(int lam, int N, double k, double r, int l_max) {
    Eigensystem sys = GetEigensystem(lam, l_max);
    if (N < 0 || N >= (int)sys.eigvals.size()) return 0.0;
    
    double eigval = sys.eigvals[N];
    
    
    double val_sq = 0.25 + eigval;
    if (val_sq < 0) return 0.0; // Bound/Complex
    
    double L_eff = -0.5 + std::sqrt(val_sq);
    double kr = k * r;
    
    if (kr < 1e-10) {
        return (L_eff < 1e-5) ? 1.0 : 0.0;
    }
    
    double bessel = MathSpecial::CylBesselJ(L_eff + 0.5, kr);
    double pre = std::sqrt(M_PI / (2.0 * kr));
    
    return pre * bessel;
}
