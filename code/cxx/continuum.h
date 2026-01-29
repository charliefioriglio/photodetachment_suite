#ifndef CONTINUUM_H
#define CONTINUUM_H

#include <complex>
#include <vector>
#include "math_special.h"

// Constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Continuum {

    // Evaluate simple Plane Wave e^(i k.r)
    // k_vec: [kx, ky, kz]
    // r_vec: [x, y, z]
    std::complex<double> EvaluatePlaneWave(const double* k_vec, const double* r_vec);

    // Evaluate Plane Wave Expansion at a point r
    // 4pi * Sum_{l,m} i^l * j_l(kr) * Y_lm(r_hat) * Y_lm^*(k_hat)
    // l_max: Maximum angular momentum to include
    std::complex<double> EvaluatePlaneWaveExpansion(const double* k_vec, const double* r_vec, int l_max);

    // Evaluate Radial Part j_l(kr)
    double EvaluateRadial(int l, double k, double r);

}

#endif
