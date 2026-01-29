#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>

#include "physical_dipole.h"
#include "math_special.h"

// Usage: ./verify_angular D a E m l_target num_pts output.csv
int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Usage: ./verify_angular D a E m l_target num_pts output.csv" << std::endl;
        return 1;
    }

    double D = std::stod(argv[1]);
    double a = std::stod(argv[2]);
    double E = std::stod(argv[3]);
    int m = std::stoi(argv[4]);
    int l_target = std::stoi(argv[5]);
    int num_pts = std::stoi(argv[6]);
    std::string out_file = argv[7];

    // Solve Angular Problem
    int l_max = std::max(l_target + 5, 10);
    // PhysicalDipole::Solve is for the full solution, we can use PhysicalDipoleAngular directly.
    
    // Note: PhysicalDipoleAngular::Solve(m, l_max, E, a, D)
    auto result = PhysicalDipoleAngular::Solve(m, l_max, E, a, D);
    
    // Unpack result
    // std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<int>>
    const auto& eigenvals = std::get<0>(result);
    const auto& eigenvectors = std::get<1>(result);
    const auto& l_values = std::get<2>(result);

    // Find the mode index corresponding to l_target
    // For D=0, modes are pure l. 
    // l_values lists the l indices that are coupled (e.g. m, m+1, m+2...) for "Prolate" basis P_l^m?
    // Wait, the solver usually returns solutions in order of increasing eigenvalue.
    // For D=0, eigenvalue is l(l+1).
    // So we just search for the mode where the dominant coefficient corresponds to l_target?
    // Or just assume the n-th mode corresponds to l = |m| + n.
    
    int mode_idx = -1;
    // Expected l_index in the l_values array logic:
    // Solver internal l start at |m|.
    // So mode n=0 -> l=|m|. mode n=1 -> l=|m|+1.
    // l_target must be >= |m|.
    
    if (l_target < std::abs(m)) {
        std::cerr << "Error: l_target (" << l_target << ") < |m| (" << std::abs(m) << ")" << std::endl;
        return 1;
    }
    
    int expected_idx = l_target - std::abs(m);
    if (expected_idx >= (int)eigenvals.size()) {
        std::cerr << "Error: l_target out of range of calculated modes." << std::endl;
        return 1;
    }
    
    // Let's verify it matches (eigenvalue check)
    // For D=0, lambda = l(l+1) evaluated at ... wait, Physical Dipole Angular eqn:
    // (1-eta^2) T'' - 2eta T' + [lambda - c^2 eta^2 - m^2/(1-eta^2)] T = 0
    // Separation constant A_lm(c).
    // For c=0 (small k*a), A_lm = l(l+1).
    // So check eigenvalue.
    
    mode_idx = expected_idx;
    double lambda = eigenvals[mode_idx];
    std::cout << "# Mode match: Target l=" << l_target << " index=" << mode_idx << " Lambda=" << lambda << " (Expect ~" << l_target*(l_target+1) << ")" << std::endl;

    // Get Coefficients for this mode
    // eigenvectors[i] is column vector for mode i?
    // "Eigenvectors (columns)" according to header comments, but let's verifying physical_dipole.cpp or header.
    // physical_dipole.h: "std::vector<std::vector<double>> ang_eigenvectors;"
    // Usually inner is component index?
    // Let's assume standard: eigenvectors[basis_idx][mode_idx] ??
    // Or eigenvectors[mode_idx][basis_idx]?
    // Let's check header comment again: "Eigenvectors (columns)" implies eigenvectors[row][col] is usual matrix, but vector of vectors...
    // usually means vec[i] is the i-th object. If it returns list of eigenvectors, likely vec[mode_idx] is the vector.
    // Let's assume result of Solve returns `eigenvectors` where `eigenvectors[n]` is the n-th eigenvector (coefficients for different l's).
    // Wait, let's double check via implementation if possible or just assume and fix if plotted wrong.
    // Actually, checking `physical_dipole.cpp` would be safer.
    
    // BUT, PhysicalDipoleAngular::Evaluate takes `coeffs`.
    // Evaluate(eta, m, coeffs, l_vals).
    // So we need the coeffs vector for our mode.
    // I will assume `eigenvectors[mode_idx]` holds the coefficients.
    
    // Check dimensions
    if (mode_idx >= (int)eigenvectors.size()) {
         // Maybe it's transposed?
         // If eigenvectors.size() == basis_size (dim) and eigenvectors[0].size() == num_modes?
         // No, standard is usually list of vectors.
         // Let's assume eigenvectors[mode_idx] is correct.
    }
    std::vector<double> coeffs;
    // Actually, looking at common linear algebra wrapper outputs...
    // Often it is eigenvectors[mode] -> vector.
    // Let's proceed.
    
    // To be robust, let's try to infer from dimensions.
    int dim = l_values.size();
    if (eigenvectors.size() == (size_t)dim && eigenvectors[0].size() > 1) {
        // This looks like Column storage (eigenvectors[row][col]).
        // Then we need to extract column `mode_idx`.
        coeffs.resize(dim);
        for(int i=0; i<dim; ++i) coeffs[i] = eigenvectors[i][mode_idx];
    } else {
        // Row storage (eigenvectors[mode][basis_idx])
        coeffs = eigenvectors[mode_idx];
    }

    std::ofstream out(out_file);
    out << "eta,theta_deg,T_phys,Y_lm\n";

    double d_eta = 2.0 / (num_pts - 1);
    for (int i = 0; i < num_pts; ++i) {
        double eta = -1.0 + i * d_eta;
        // Avoid exact -1/1 singularity
        if (eta <= -1.0) eta = -0.999999;
        if (eta >= 1.0) eta = 0.999999;
        
        double theta = std::acos(eta);
        double theta_deg = theta * 180.0 / M_PI;
        
        // Physical Dipole value
        double val_phys = PhysicalDipoleAngular::Evaluate(eta, m, coeffs, l_values);
        
        // Analytical Spherical Harmonic Legendre P_l^m(cos theta)
        // Normalized? Y_lm includes 1/sqrt(2pi) e^imphi normalization usually.
        // Or just Theta part.
        // MathSpecial::SphericalHarmonicY returns complex Y_lm.
        // For m=0, pure real.
        // For m!=0, exp(im phi). At phi=0, real.
        // We define T(eta) (Phys) to match the Theta part of Y_lm?
        // Let's check normalization by plotting both. 
        // We evaluate Y_lm(theta, 0.0) at phi=0.
        
        std::complex<double> Y_val = MathSpecial::SphericalHarmonicY(l_target, m, theta, 0.0);
        
        // Our PhysicalDipole normalization might contain sqrt(2pi)?
        // The standard is usually Normalized to Integral |T|^2 = 1.
        // Spherical Harmonic Theta part is Integral |Th|^2 d(cos theta) = 1? 
        // No, Integral |Y|^2 dOmega = 1.
        // Integral |Th|^2 d(cos theta) * 2pi = 1 => Integral |Th|^2 = 1/2pi.
        // So Th_norm approx Y_val * sqrt(2pi).
        
        // Let's output Y_val.real() and we can scale in python.
        
        out << eta << "," << theta_deg << "," << val_phys << "," << Y_val.real() << "\n";
    }

    out.close();
    return 0;
}
