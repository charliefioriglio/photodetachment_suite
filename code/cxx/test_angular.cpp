#include "physical_dipole.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: ./test_angular m l_max E a D [output_file]" << std::endl;
        return 1;
    }
    
    int m = std::stoi(argv[1]);
    int l_max = std::stoi(argv[2]);
    double E = std::stod(argv[3]);
    double a = std::stod(argv[4]);
    double D = std::stod(argv[5]);
    
    auto [eigenvalues, eigenvectors, ell_vals] = PhysicalDipoleAngular::Solve(m, l_max, E, a, D);
    
    std::cout << "Computed " << eigenvalues.size() << " eigenvalues." << std::endl;
    for(size_t i=0; i<std::min(eigenvalues.size(), (size_t)10); ++i) {
        std::cout << "Eig " << i << ": " << std::fixed << std::setprecision(8) << eigenvalues[i] << std::endl;
    }
    
    if (argc >= 7) {
        std::string outfile = argv[6];
        std::ofstream out(outfile);
        out << "Eta";
        for(size_t k=0; k<std::min(eigenvalues.size(), (size_t)5); ++k) {
            out << ",State_" << k;
        }
        out << "\n";
        
        // Grid for eta: -1 to 1
        int steps = 200;
        for(int i=0; i<=steps; ++i) {
            double eta = -1.0 + 2.0 * i / steps;
            
            out << eta;
            // Print top 5 states
            for(size_t k=0; k<std::min(eigenvalues.size(), (size_t)5); ++k) {
                // Determine coeff vector for state k
                std::vector<double> coeffs;
                for(size_t row=0; row<eigenvectors.size(); ++row) {
                    coeffs.push_back(eigenvectors[row][k]);
                }
                double val = PhysicalDipoleAngular::Evaluate(eta, m, coeffs, ell_vals);
                out << "," << val;
            }
            out << "\n";
        }
    }
    
    return 0;
}
