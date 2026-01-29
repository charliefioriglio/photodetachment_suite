#include "continuum.h"
#include <cmath>
#include <iostream>

namespace Continuum {

    std::complex<double> EvaluatePlaneWave(const double* k_vec, const double* r_vec) {
        double dot = k_vec[0] * r_vec[0] + k_vec[1] * r_vec[1] + k_vec[2] * r_vec[2];
        return std::exp(std::complex<double>(0.0, dot));
    }

    std::complex<double> EvaluatePlaneWaveExpansion(const double* k_vec, const double* r_vec, int l_max) {
        double r = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);
        double k = std::sqrt(k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2]);
        double kr = k * r;
        
        // Angles for r_vec
        double theta_r = (r > 1e-12) ? std::acos(r_vec[2] / r) : 0.0;
        double phi_r = (r > 1e-12) ? std::atan2(r_vec[1], r_vec[0]) : 0.0;
        
        // Angles for k_vec
        double theta_k = (k > 1e-12) ? std::acos(k_vec[2] / k) : 0.0;
        double phi_k = (k > 1e-12) ? std::atan2(k_vec[1], k_vec[0]) : 0.0;
        
        std::complex<double> sum(0.0, 0.0);
        std::complex<double> i_unit(0.0, 1.0);
        
        for (int l = 0; l <= l_max; ++l) {
            // j_l(kr)
            double jl = MathSpecial::SphericalBesselJ(l, kr);
            
            // i^l
            std::complex<double> i_pow_l = std::pow(i_unit, l); // Or explicitly handle 1, i, -1, -i
            
            for (int m = -l; m <= l; ++m) {
                // Y_lm(r) and Y_lm(k)^*
                std::complex<double> Y_r = MathSpecial::SphericalHarmonicY(l, m, theta_r, phi_r);
                std::complex<double> Y_k = MathSpecial::SphericalHarmonicY(l, m, theta_k, phi_k);
                
                sum += i_pow_l * jl * Y_r * std::conj(Y_k);
            }
        }
        
        return 4.0 * M_PI * sum;
    }

    double EvaluateRadial(int l, double k, double r) {
        return MathSpecial::SphericalBesselJ(l, k * r);
    }

}
