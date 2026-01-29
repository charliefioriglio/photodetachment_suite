#ifndef MATH_SPECIAL_H
#define MATH_SPECIAL_H

#include <complex>
#include <vector>

namespace MathSpecial {

    // Generic Cylinder Bessel J for fractional order nu
    double CylBesselJ(double nu, double x);

    // Spherical Bessel J_l(x) = sqrt(pi/2x) J_{l+1/2}(x)
    double SphericalBesselJ(int l, double x);

    // Generic Spherical Bessel j_nu(x) for non-integer nu
    double SphericalBesselJGeneric(double nu, double x);

    // Spherical Bessel j_nu(x) for complex nu
    std::complex<double> SphericalBesselJComplex(std::complex<double> nu, double x);

    // Spherical Harmonic Y_lm(theta, phi)
    // Returns complex value Y_lm
    std::complex<double> SphericalHarmonicY(int l, int m, double theta, double phi);

}

#endif
