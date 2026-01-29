#include "point_dipole.h"
#include "continuum.h"
#include "math_special.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Usage: ./continuum_plotter [radial|angular] [D] [Energy_au] [L_max] [output_file]

void ScanRadial(double D, double E_au, int l_max, const std::string& filename) {
    std::ofstream out(filename);
    out << "r,Re_Analytic,Im_Analytic,Re_PWE,Im_PWE,Re_PD,Im_PD\n";
    
    double k = std::sqrt(2.0 * E_au);
    double k_vec[3] = {0.0, 0.0, k}; // k along z
    
    // Scan along z axis (theta=0)
    // Actually, let's scan along x (theta=90) to see angular structure, or z to see radial?
    // Point Dipole with k||z, m=0 is excited.
    // Along z (theta=0), Y_lm is non-zero only for m=0.
    // Along x (theta=90), Y_lm varies.
    // Let's do along Z for now (classic radial).
    
    double x = 0.0;
    double y = 0.0;
    
    PointDipole pd(D);
    
    for (int i=1; i<=1000; ++i) {
        double r = 0.1 * i; // 0.1 to 100.0
        double z = r; 
        double r_vec[3] = {x, y, z};
        
        // 1. Analytic Plane Wave
        std::complex<double> psi_ana = Continuum::EvaluatePlaneWave(k_vec, r_vec);
        
        // 2. Reference PWE
        std::complex<double> psi_pwe = Continuum::EvaluatePlaneWaveExpansion(k_vec, r_vec, l_max);
        
        // 3. Point Dipole
        std::complex<double> psi_pd = pd.EvaluateDirectional(k_vec, E_au, x, y, z, l_max);
        
        out << r << "," 
            << psi_ana.real() << "," << psi_ana.imag() << ","
            << psi_pwe.real() << "," << psi_pwe.imag() << ","
            << psi_pd.real() << "," << psi_pd.imag() << "\n";
    }
    out.close();
}

void ScanAngular(double D, double E_au, int l_max, const std::string& filename) {
    std::ofstream out(filename);
    out << "theta,Re_Analytic,Im_Analytic,Re_PWE,Im_PWE,Re_PD,Im_PD\n";
    
    double k = std::sqrt(2.0 * E_au);
    double k_vec[3] = {0.0, 0.0, k}; // k along z
    
    double r = 10.0; // Fixed radius
    PointDipole pd(D);
    
    for (int i=0; i<=180; ++i) {
        double theta_deg = (double)i;
        double theta = theta_deg * M_PI / 180.0;
        
        // Scan in x-z plane (phi=0)
        double z = r * std::cos(theta);
        double x = r * std::sin(theta);
        double y = 0.0;
        double r_vec[3] = {x, y, z};
        
        // 1. Analytic
        std::complex<double> psi_ana = Continuum::EvaluatePlaneWave(k_vec, r_vec);
        
        // 2. Reference PWE
        std::complex<double> psi_pwe = Continuum::EvaluatePlaneWaveExpansion(k_vec, r_vec, l_max);
        
        // 3. Point Dipole
        std::complex<double> psi_pd = pd.EvaluateDirectional(k_vec, E_au, x, y, z, l_max);
        
        out << theta_deg << "," 
            << psi_ana.real() << "," << psi_ana.imag() << ","
            << psi_pwe.real() << "," << psi_pwe.imag() << ","
            << psi_pd.real() << "," << psi_pd.imag() << "\n";
    }
    out.close();
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " [radial|angular] [D] [Energy_au] [L_max] [output_file]" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    double D = std::stod(argv[2]);
    double E = std::stod(argv[3]);
    int L = std::stoi(argv[4]);
    std::string out = argv[5];
    
    if (mode == "radial") {
        ScanRadial(D, E, L, out);
    } else if (mode == "angular") {
        ScanAngular(D, E, L, out);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
    
    return 0;
}
