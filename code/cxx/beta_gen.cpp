#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "tools.h"
#include "molecule.h"
#include "dyson.h"
#include "num_eikr.h"
#include "grid.h"
#include "angle_grid.h"
#include "grid.h"
#include "angle_grid.h"
#include "beta.h"
#include "cross_section.h" // Includes PhysicalDipole support in Calculator

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input_file output_file [--beta]" << std::endl;
        // Output file arg is kept to match dyson_gen signature but maybe unused or used for beta.csv
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2]; // Unused for Beta, but arguments must match python call structure?
    // Actually, python doesn't call this yet. I will call it manually.
    
    std::ifstream in(input_file);
    if (!in) {
        std::cerr << "Error: Could not open input file " << input_file << std::endl;
        return 1;
    }

    Molecule mol;
    int n_atoms;
    if (!(in >> n_atoms)) return 1;

    for (int i = 0; i < n_atoms; ++i) {
        std::string sym;
        int idx;
        double x, y, z;
        in >> sym >> idx >> x >> y >> z;
        mol.add_atom(sym, idx, x, y, z);
    }
    
    int n_shells;
    in >> n_shells;
    for (int i = 0; i < n_shells; ++i) {
        int atom_idx, l, n_prim;
        bool is_pure;
        in >> atom_idx >> l >> is_pure >> n_prim;
        
        std::vector<double> exps(n_prim);
        std::vector<double> coeffs(n_prim);
        for (int j = 0; j < n_prim; ++j) {
            in >> exps[j] >> coeffs[j];
        }
        mol.add_shell_to_atom(atom_idx, l, is_pure, exps, coeffs);
    }
    
    int num_dyson_orbs;
    if (!(in >> num_dyson_orbs)) num_dyson_orbs = 1; 

    std::vector<Dyson> dysons;
    for (int d = 0; d < num_dyson_orbs; ++d) {
        int n_coeffs;
        double norm_val = 1.0;
        in >> n_coeffs >> norm_val; 
        std::vector<double> coeffs(n_coeffs);
        for (int i = 0; i < n_coeffs; ++i) in >> coeffs[i];
        
        std::string label = (d==0) ? "Left" : "Right";
        Dyson d_obj(&mol, coeffs, label);
        d_obj.qchem_norm = norm_val;
        dysons.push_back(d_obj);
    }
    
    double x0, x1, y0, y1, z0, z1, step;
    in >> x0 >> x1 >> y0 >> y1 >> z0 >> z1 >> step;
    
    // Renormalize
    for(auto& do_obj : dysons) {
        do_obj.renormalize(x0, x1, y0, y1, z0, z1, step);
    }
    
    // Make Grid for integration
    UniformGrid grid(x0, x1, y0, y1, z0, z1, step);

    // Default Beta Energies
    std::vector<double> beta_energies = {
        0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5
    };
    
    int n_points = 150; // Default (Hardcoded)
    bool use_hardcoded = true;

    // Parse args
    // Parse args
    bool use_pwe = false;
    for(int i=3; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--energies") {
            beta_energies.clear();
            int j = i + 1;
            while(j < argc) {
                std::string val = argv[j];
                if (val.substr(0, 2) == "--") break; 
                try {
                    beta_energies.push_back(std::stod(val));
                } catch(...) {
                    break; 
                }
                j++;
            }
            i = j - 1;
        } else if (arg == "--points") {
            if (i + 1 < argc) {
                try {
                    n_points = std::stoi(argv[i+1]);
                    // use_hardcoded = false; // Removed to allow explicit 150 to use hardcoded
                    i++;
                } catch (...) {}
            }
        } else if (arg == "--pwe") {
            use_pwe = true;
        }
    }
    
    if (beta_energies.empty()) {
        std::cerr << "Error: No energies specified." << std::endl;
        return 1;
    }

    AngleGrid angle_grid;
    if (use_hardcoded && n_points == 150) {
        std::cerr << "Generating Hardcoded Angle Grid (150 pts)..." << std::endl;
        angle_grid.GenerateHardcoded();
    } else {
        std::cerr << "Generating Repulsion Angle Grid (" << n_points << " pts)..." << std::endl;
        angle_grid.GenerateRepulsion(n_points);
    }
    
    const Dyson& L_orig = dysons[0];
    const Dyson& R_orig = (dysons.size() > 1) ? dysons[1] : dysons[0];

    // Centering Logic (User Request)
    // 1. Calculate Centroid using the bounding box of the grid
    std::cout << "Calculating Dyson centroid..." << std::endl;
    // Use the grid parameters from input for bounding box
    Dyson::Vector3 centroid = L_orig.get_centroid(grid.xmin, grid.xmax, grid.ymin, grid.ymax, grid.zmin, grid.zmax, grid.dx);
    
    std::cout << "Dyson Centroid: (" << centroid.x << ", " << centroid.y << ", " << centroid.z << ")" << std::endl;
    std::cout << "Shifting molecule to center Dyson orbital at (0,0,0)..." << std::endl;
    
    // 2. Shift Molecule
    mol.shift_geometry(-centroid.x, -centroid.y, -centroid.z);
    
    // 3. Update Dyson Basis Caches
    for(auto& d : dysons) {
        d.update_geometry();
    }
    
    // Get updated references
    const Dyson& L = dysons[0];
    const Dyson& R = (dysons.size() > 1) ? dysons[1] : dysons[0];

    // Results container
    struct ResultRow {
        double e;
        double par;
        double perp;
        double beta;
    };
    std::vector<ResultRow> final_results;

    int l_max = 3;
    bool explicit_lmax = false;
    bool use_point_dipole = false;
    bool use_physical_dipole = false;
    double dipole_strength = 0.0;
    double dipole_length = 0.0; // 'a' parameter for Physical Dipole

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pwe") {
            use_pwe = true;
        } else if (arg == "--points" && i + 1 < argc) {
             std::string points_str = argv[++i];
             try {
                 n_points = std::stoi(points_str);
             } catch(...) {
                 std::cerr << "Invalid number of points: " << points_str << std::endl;
                 return 1;
             }
        } else if (arg == "--lmax" && i + 1 < argc) {
             std::string lmax_str = argv[++i];
             try {
                 l_max = std::stoi(lmax_str);
                 explicit_lmax = true;
             } catch(...) {
                 std::cerr << "Invalid lmax: " << lmax_str << std::endl;
                 return 1;
             }
        } else if (arg == "--point-dipole" && i + 1 < argc) {
             try {
                 dipole_strength = std::stod(argv[++i]);
                 use_point_dipole = true;
             } catch(...) {
                 std::cerr << "Invalid dipole strength" << std::endl;
                 return 1;
             }
        } else if (arg == "--physical-dipole" && i + 2 < argc) {
             // Expects: --physical-dipole D a
             try {
                 dipole_strength = std::stod(argv[++i]);
                 dipole_length = std::stod(argv[++i]);
                 use_physical_dipole = true;
             } catch(...) {
                 std::cerr << "Invalid physical dipole params" << std::endl;
                 return 1;
             }
        }
    }

    if (use_pwe) {
        std::cout << "Calculating Beta parameters using Plane Wave Expansion (Analytic Averaging)..." << std::endl;
        std::cout << "Using l_max = " << l_max << std::endl;
        auto beta_results = BetaCalculator::CalculateBetaAnalytic(L, R, grid, beta_energies, l_max);
        for(const auto& res : beta_results) {
            final_results.push_back({res.energy, res.sigma_par, res.sigma_perp, res.beta});
        }
    } else if (use_physical_dipole) {
        std::cout << "Calculating Cross Section using Physical Dipole Model (D=" << dipole_strength << ", a=" << dipole_length << ")..." << std::endl;
        std::cout << "Using l_max = " << l_max << std::endl;
        
        // Determine Dipole Axis (from atoms 0 and 1 if available)
        std::vector<double> dipole_axis = {0.0, 0.0, 1.0};
        std::vector<double> dipole_center = {0.0, 0.0, 0.0};
        if (mol.atoms.size() >= 2) {
            double dx = mol.atoms[1].x - mol.atoms[0].x;
            double dy = mol.atoms[1].y - mol.atoms[0].y;
            double dz = mol.atoms[1].z - mol.atoms[0].z;
            dipole_axis = {dx, dy, dz};
            
            dipole_center[0] = 0.5 * (mol.atoms[0].x + mol.atoms[1].x);
            dipole_center[1] = 0.5 * (mol.atoms[0].y + mol.atoms[1].y);
            dipole_center[2] = 0.5 * (mol.atoms[0].z + mol.atoms[1].z);
            
            std::cout << "Bond Axis: (" << dx << ", " << dy << ", " << dz << ")" << std::endl;
             std::cout << "Bond Center: (" << dipole_center[0] << ", " << dipole_center[1] << ", " << dipole_center[2] << ")" << std::endl;
        }


        // ComputePhysicalDipoleCrossSection returns vector<double> (Total Sigma only)
        // Beta calculation requires parallel/perp components or full m-averaging not yet exposed in single function?
        // Wait, CrossSectionCalculator::ComputePhysicalDipoleCrossSection returns only Sigma Total.
        // It does NOT calculate Beta parameters (par/perp).
        // For Verification of Cross Section, this is sufficient. 
        // We will output 0 for beta/par/perp if not available, or just Sigma.
        // The output CSV expects par, perp, beta.
        
        // We need to fetch Photon Energies. beta_energies are eKE.
        // We need IE. Where is IE? 
        // beta_gen doesn't take IE input? 
        // NumEikr uses eKE directly.
        // CrossSectionCalculator usually takes Photon Energies + IE.
        // It converts Photon - IE -> eKE.
        // Here we have eKE. So we can fake IE=0 and pass eKE as Photon Energy.
        
        CrossSectionCalculator calc;
        auto sigma_results = calc.ComputePhysicalDipoleCrossSection(
            L, R, grid, beta_energies, 0.0, l_max, dipole_strength, dipole_length, dipole_axis, dipole_center
        );
        
        for(size_t k=0; k<beta_energies.size(); ++k) {
             // We don't have par/perp/beta, just Total.
             // We can put Total in 'par' column or just make a special output?
             // The output format is rigid: eKE,Par,Perp,Beta.
             // We can put Sigma in Par, 0 in Perp, 0 in Beta?
             // Or better: Sigma in 'par' (Total) and others 0.
             // This is a hack for verification.
             final_results.push_back({beta_energies[k], sigma_results[k], 0.0, 0.0});
        }
        
    } else if (use_point_dipole) {
        std::cout << "Calculating Beta parameters using Point Dipole Model (D=" << dipole_strength << ")..." << std::endl;
        std::cout << "Using l_max = " << l_max << std::endl;
        
        auto beta_results = BetaCalculator::CalculateBetaPointDipole(L, R, grid, beta_energies, dipole_strength, l_max);
        
        for(const auto& res : beta_results) {
            final_results.push_back({res.energy, res.sigma_par, res.sigma_perp, res.beta});
        }
    } else {
        // Use NumEikr for optimized calculation (ezDyson logic)
        std::cout << "Calculating Beta parameters using NumEikr (ezDyson algorithm)..." << std::endl;
        NumEikr num_eikr;
        num_eikr.compute(grid, L, R, angle_grid, beta_energies);
        
        for(size_t i=0; i<beta_energies.size(); ++i) {
            double eKE = beta_energies[i];
            double par = num_eikr.get_sigma_par(i);
            double perp = num_eikr.get_sigma_perp(i);
            
            // Beta formula: 2(Par - Perp) / (Par + 2*Perp)
            double denom = par + 2.0 * perp;
            double beta = 0.0;
            if (std::abs(denom) > 1e-14) {
                beta = 2.0 * (par - perp) / denom;
            }
            final_results.push_back({eKE, par, perp, beta});
        }
    }
    
    // Write Results
    std::ofstream beta_file(output_file);
    beta_file << "eKE,SigmaPar,SigmaPerp,Beta\n";
    
    for(const auto& row : final_results) {
        beta_file << row.e << "," << row.par << "," << row.perp << "," << row.beta << "\n";
    }
    beta_file.close();
    
    std::cout << "Done. Written to " << output_file << std::endl;
    
    return 0;
}
