#include "physical_dipole.h"
#include "math_special.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

int main() {
    std::cout << "=== Test Radial Solver ===" << std::endl;
    
    // Case 1: Free space (c=0, D=0)
    // Should recover spherical Bessel functions j_l
    // nu should be integer l (or -l-1)
    
    double a = 1.0;
    double E = 0.5; // k=1
    double c = std::sqrt(2*E)*a; // 1.0 * 1.0 = 1.0
    // Try c -> 0
    c = 1e-4;
    int m = 0;
    
    std::cout << "\n--- Limit c -> 0, D -> 0 (Free Space) ---" << std::endl;
    // For free space, eigenvalues are l(l+1).
    // Our solver uses Alm = -lambda.
    // So if lambda = -l(l+1), Alm = l(l+1).
    
    for (int l = 0; l <= 2; ++l) {
        double Alm = (double)(l * (l + 1));
        std::complex<double> nu = PhysicalDipoleRadial::SolveNu(c, m, Alm);
        
        std::cout << "l=" << l << ", Alm=" << Alm << " -> nu=" << nu << std::endl;
        
        // Coeffs
        auto coeffs = PhysicalDipoleRadial::ComputeCoefficients(nu, c, m, Alm);
        std::cout << "Coeffs size: " << coeffs.size() << std::endl;
        for(auto const& [L, val] : coeffs) {
            if(std::abs(val) > 1e-4)
                std::cout << "  L=" << L << ": " << val << std::endl;
        }
    }
    
    // Case 2: Physical Dipole (Non-zero c, D)
    std::cout << "\n--- Physical Dipole c=1.0, D=0.6 ---" << std::endl;
    c = 1.0;
    double D = 0.6;
    // We need eigenvalue lambda first.
    // Approximate lambda using Point Dipole or just Angular Solver?
    // Let's use Angular Solver to get real eigenvalue.
    auto ang_res = PhysicalDipoleAngular::Solve(m, 5, E, a, D);
    std::vector<double> evals = std::get<0>(ang_res);
    
    for (int i = 0; i < std::min((int)evals.size(), 3); ++i) {
        double lambda = evals[i];
        double Alm = -lambda;
        std::cout << "Mode " << i << ": lambda=" << lambda << " -> Alm=" << Alm << std::endl;
        
        std::complex<double> nu = PhysicalDipoleRadial::SolveNu(c, m, Alm);
        std::cout << "  nu = " << nu << std::endl;
        
        auto coeffs = PhysicalDipoleRadial::ComputeCoefficients(nu, c, m, Alm);
        std::cout << "  Dominant Coeffs:" << std::endl;
        for(auto const& [L, val] : coeffs) {
             if(std::abs(val) > 0.1)
                std::cout << "    L=" << L << ": " << val << std::endl;
        }
    }

    return 0;
}
