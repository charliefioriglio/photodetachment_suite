#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "tools.h"
#include "molecule.h"
#include "dyson.h"
#include "grid.h"
#include "cross_section.h"

// Usage: ./compute_xs input_file D a energies(comma_sep) [IE]
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " input_file D a energies [IE]" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    double D = std::stod(argv[2]);
    double a = std::stod(argv[3]);
    std::string energies_str = argv[4];
    
    double IE = 1.68; 
    if (argc >= 6) IE = std::stod(argv[5]);

    // Parse Energies
    std::vector<double> energies;
    std::stringstream ss(energies_str);
    std::string item;
    while(std::getline(ss, item, ',')) {
        energies.push_back(std::stod(item));
    }
    
    // Parse Input File (Copy of dyson_gen logic)
    std::ifstream in(input_file);
    if (!in) {
        std::cerr << "Error: Could not open input file " << input_file << std::endl;
        return 1;
    }

    Molecule mol;
    int n_atoms;
    in >> n_atoms;
    for (int i = 0; i < n_atoms; ++i) {
        std::string sym; int idx; double x, y, z;
        in >> sym >> idx >> x >> y >> z;
        mol.add_atom(sym, idx, x, y, z);
    }
    
    int n_shells;
    in >> n_shells;
    for (int i = 0; i < n_shells; ++i) {
        int atom_idx, l, n_prim; bool is_pure;
        in >> atom_idx >> l >> is_pure >> n_prim;
        std::vector<double> exps(n_prim);
        std::vector<double> coeffs(n_prim);
        for (int j = 0; j < n_prim; ++j) in >> exps[j] >> coeffs[j];
        mol.add_shell_to_atom(atom_idx, l, is_pure, exps, coeffs);
    }
    
    int num_dyson_orbs;
    if (!(in >> num_dyson_orbs)) num_dyson_orbs = 1;
    
    std::vector<Dyson> dysons;
    for (int d = 0; d < num_dyson_orbs; ++d) {
        int n_coeffs; double norm_val = 1.0;
        in >> n_coeffs >> norm_val;
        std::vector<double> coeffs(n_coeffs);
        for (int i = 0; i < n_coeffs; ++i) in >> coeffs[i];
        Dyson d_obj(&mol, coeffs, (d==0?"L":"R"));
        d_obj.qchem_norm = norm_val;
        dysons.push_back(d_obj);
    }
    
    // Grid Params
    double x0, x1, y0, y1, z0, z1, step;
    in >> x0 >> x1 >> y0 >> y1 >> z0 >> z1 >> step;
    
    // Centering Logic
    const Dyson& L_orig = dysons[0];
    Dyson::Vector3 centroid = L_orig.get_centroid(x0, x1, y0, y1, z0, z1, step);
    mol.shift_geometry(-centroid.x, -centroid.y, -centroid.z);
    for(auto& d_obj : dysons) d_obj.update_geometry();
    
    x0 -= centroid.x; x1 -= centroid.x;
    y0 -= centroid.y; y1 -= centroid.y;
    z0 -= centroid.z; z1 -= centroid.z;
    
    for(auto& do_obj : dysons) do_obj.renormalize(x0, x1, y0, y1, z0, z1, step);
    
    UniformGrid grid(x0, x1, y0, y1, z0, z1, step);
    
    // Calculate XS
    Dyson& dL = dysons[0];
    Dyson& dR = (dysons.size() > 1) ? dysons[1] : dysons[0];
    
    CrossSectionCalculator calc;
    // l_max = 5 ?
    int l_max = 5;
    
    std::cout << "Calculating Physical Dipole XS (D=" << D << ", a=" << a << ")..." << std::endl;
    
    // Determine Dipole Axis from atoms
    std::vector<double> dipole_axis = {0.0, 0.0, 1.0};
    if (mol.atoms.size() >= 2) {
        // Assume first two atoms define the axis (usually 0 and 1)
        double dx = mol.atoms[1].x - mol.atoms[0].x;
        double dy = mol.atoms[1].y - mol.atoms[0].y;
        double dz = mol.atoms[1].z - mol.atoms[0].z;
        dipole_axis = {dx, dy, dz};
        std::cout << "Detected Dipole Axis: (" << dx << ", " << dy << ", " << dz << ")" << std::endl;
    } else {
        std::cout << "Warning: < 2 atoms. Using default Z-axis." << std::endl;
    }
    
    // Determine Dipole Center (Midpoint of bond)
    std::vector<double> dipole_center = {0.0, 0.0, 0.0};
    /*
    if (mol.atoms.size() >= 2) {
        dipole_center[0] = 0.5 * (mol.atoms[0].x + mol.atoms[1].x);
        dipole_center[1] = 0.5 * (mol.atoms[0].y + mol.atoms[1].y);
        dipole_center[2] = 0.5 * (mol.atoms[0].z + mol.atoms[1].z);
        std::cout << "Detected Dipole Center (shifted): (" 
                  << dipole_center[0] << ", " << dipole_center[1] << ", " << dipole_center[2] << ")" << std::endl;
    }
    */

    auto results = calc.ComputePhysicalDipoleCrossSection(
        dL, dR, grid, energies, IE, l_max, D, a, dipole_axis, dipole_center
    );
    
    std::cout << "Energy_eV,CrossSection\n";
    for(size_t i=0; i<energies.size(); ++i) {
        std::cout << energies[i] << "," << results[i] << "\n";
    }
    
    // Also run PWE for comparison
    std::cout << "Calculating PWE XS..." << std::endl;
    auto results_pwe = calc.ComputeTotalCrossSection(
        dL, dR, grid, energies, IE, l_max
    );
     std::cout << "Energy_eV,CrossSection_PWE\n";
    for(size_t i=0; i<energies.size(); ++i) {
        std::cout << energies[i] << "," << results_pwe[i] << "\n";
    }

    return 0;
}
