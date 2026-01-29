#include "shell.h"
#include <cmath>

Shell::Shell(int angular_mom, bool pure, const std::vector<double>& exps, const std::vector<double>& coeffs)
    : l(angular_mom), is_pure(pure), exponents(exps), coefficients(coeffs) {
    build();
}

void Shell::build() {
    contracted_functions.clear();
    
    // Determine the list of functions to build (Cartesian or Pure)
    // Each function is a linear combination of ContractedGaussians (which are just sums of primitives)
    // IMPORTANT: Wait, the standard "Contracted Gaussian" is usually one set of lx,ly,lz
    // For pure functions, we need linear combinations of DIFFERENT lx,ly,lz cartesian contractions
    // So 'contracted_functions' should probably be a more complex object if we want to store it fully formed
    
    // Actually, in the standard AO basis logic:
    // A single basis function mu is:
    // phi_mu = Sum(coeff * primitive(exp))  <-- Standard contraction
    // But for Pure functions:
    // phi_pure = Sum(pure_coeff * phi_cartesian)
    // And phi_cartesian is a contraction of primitives.
    
    // So we'll flatten this.
    // We will generate the needed Cartesian contractions first.
    
    if (!is_pure || l < 2) {
        auto monomials = get_cartesian_monomials(l);
        for (const auto& mon : monomials) {
            ContractedGaussian cg(mon[0], mon[1], mon[2]);
            for (size_t i = 0; i < exponents.size(); ++i) {
                cg.add_primitive(exponents[i], coefficients[i]);
            }
            cg.normalize();
            contracted_functions.push_back(cg);
        }
    } else {
        auto pure_funcs = get_pure_functions(l);
        for (const auto& pf : pure_funcs) {
            // A pure function is a combination of cartesian components
            // We can represent this as a sum of ContractedGaussians
            // But our ContractedGaussian class currently only holds ONE set of lx,ly,lz
            // Wait, ContractedGaussian holds a vector of Primitives.
            // Primitives have their own lx,ly,lz.
            // So a Pure Function can be represented as a ContractedGaussian where the primitives have different lx,ly,lz!
            // Yes!
            
            // We can just create a "Super" ContractedGaussian for the pure function
            // It just has 0,0,0 as nominal lx,ly,lz (doesn't matter)
            // And contains all the primitives from all component cartesian gaussians
            
            ContractedGaussian pure_cg(0,0,0); 
            
            for (const auto& comp : pf.components) {
                double pure_weight = comp.first;
                int lx = comp.second[0];
                int ly = comp.second[1];
                int lz = comp.second[2];
                
                // For this Cartesian component, we have the standard contraction
                // We need to add all its primitives to the pure_cg, scaled by pure_weight
                
                // First, build a temporary standard cartesian to get normalized coefficients
                ContractedGaussian temp_cart(lx, ly, lz);
                for (size_t i = 0; i < exponents.size(); ++i) {
                    temp_cart.add_primitive(exponents[i], coefficients[i]);
                }
                temp_cart.normalize();
                
                // Now add these normalized primitives to the pure_cg
                for (const auto& prim : temp_cart.primitives) {
                // primitive constructor: alpha, lx, ly, lz, coef
                // New coef = prim.coef * pure_weight
                pure_cg.primitives.emplace_back(prim.alpha, lx, ly, lz, prim.coef * pure_weight);
            }
        }
        pure_cg.normalize(); // Normalize the constructed pure function
        contracted_functions.push_back(pure_cg);
    }
    }
}

std::vector<std::vector<int>> Shell::get_cartesian_monomials(int l) {
    std::vector<std::vector<int>> combos;
    for (int lx = l; lx >= 0; --lx) {
        for (int ly = l - lx; ly >= 0; --ly) {
            int lz = l - lx - ly;
            combos.push_back({lx, ly, lz});
        }
    }
    return combos;
}

std::vector<PureFunction> Shell::get_pure_functions(int l) {
    if (l == 2) return get_pure_d();
    if (l == 3) return get_pure_f();
    if (l == 4) return get_pure_g();
    return {}; // Should error out really
}

// Coefficients transcribed from notebook/lab notebook
std::vector<PureFunction> Shell::get_pure_d() {
    std::vector<PureFunction> funcs;
    // dxy, dyz, dz2, dxz, dx2-y2 (Matching Q-Chem/aobasis.C order)
    funcs.push_back({ {{1.0, {1,1,0}}} }); // dxy
    funcs.push_back({ {{1.0, {0,1,1}}} }); // dyz
    funcs.push_back({ {{1.0, {0,0,2}}, {-0.5, {2,0,0}}, {-0.5, {0,2,0}}} }); // dz2
    funcs.push_back({ {{1.0, {1,0,1}}} }); // dxz
    funcs.push_back({ {{std::sqrt(3.0)/2.0, {2,0,0}}, {-std::sqrt(3.0)/2.0, {0,2,0}}} }); // dx2-y2
    return funcs;
}

std::vector<PureFunction> Shell::get_pure_f() {
    std::vector<PureFunction> funcs;
    // f1 (y(3x2-y2)), f2 (xyz), f3 (y(5z2-r2)), f4 (z(5z2-3r2)), f5 (x(5z2-r2)), f6 (z(x2-y2)), f7 (x(x2-3y2)) check order
    // Order from notebook:
    // 0: sqrt(5/8)*3 (2,1,0), -sqrt(5/8) (0,3,0)
    funcs.push_back({ {{3.0*std::sqrt(5.0/8.0), {2,1,0}}, {-std::sqrt(5.0/8.0), {0,3,0}}} });
    // 1: (1,1,1)
    funcs.push_back({ {{1.0, {1,1,1}}} });
    // 2: 4*sqrt(3/8) (0,1,2) - sqrt(3/8) (0,3,0) - sqrt(3/8) (2,1,0)
    funcs.push_back({ {{4.0*std::sqrt(3.0/8.0), {0,1,2}}, {-std::sqrt(3.0/8.0), {0,3,0}}, {-std::sqrt(3.0/8.0), {2,1,0}}} });
    // 3: (0,0,3) - 1.5 (2,0,1) - 1.5 (0,2,1)
    funcs.push_back({ {{1.0, {0,0,3}}, {-1.5, {2,0,1}}, {-1.5, {0,2,1}}} });
    // 4: 4*sqrt(3/8) (1,0,2) - sqrt(3/8) (3,0,0) - sqrt(3/8) (1,2,0)
    funcs.push_back({ {{4.0*std::sqrt(3.0/8.0), {1,0,2}}, {-std::sqrt(3.0/8.0), {3,0,0}}, {-std::sqrt(3.0/8.0), {1,2,0}}} });
    // 5: sqrt(15)/2 (2,0,1) - sqrt(15)/2 (0,2,1)
    funcs.push_back({ {{std::sqrt(15.0)/2.0, {2,0,1}}, {-std::sqrt(15.0)/2.0, {0,2,1}}} });
    // 6: sqrt(5/8) (3,0,0) - 3*sqrt(5/8) (1,2,0)
    funcs.push_back({ {{std::sqrt(5.0/8.0), {3,0,0}}, {-3.0*std::sqrt(5.0/8.0), {1,2,0}}} });
    
    return funcs;
}

std::vector<PureFunction> Shell::get_pure_g() {
    std::vector<PureFunction> funcs;
    // 1
    funcs.push_back({ {{std::sqrt(35.0)/2.0, {3,1,0}}, {-std::sqrt(35.0)/2.0, {1,3,0}}} });
    // 2
    funcs.push_back({ {{3.0*std::sqrt(35.0/8.0), {2,1,1}}, {-std::sqrt(35.0/8.0), {0,3,1}}} });
    // 3
    funcs.push_back({ {{3.0*std::sqrt(5.0), {1,1,2}}, {-std::sqrt(5.0)/2.0, {3,1,0}}, {-std::sqrt(5.0)/2.0, {1,3,0}}} });
    // 4
    funcs.push_back({ {{4.0*std::sqrt(5.0/8.0), {1,0,3}}, {-3.0*std::sqrt(5.0/8.0), {3,0,1}}, {-3.0*std::sqrt(5.0/8.0), {1,2,1}}} });
    // 5
    funcs.push_back({ 
        {{1.0, {0,0,4}}, {3.0/8.0, {4,0,0}}, {3.0/8.0, {0,4,0}}, 
         {-3.0, {2,0,2}}, {-3.0, {0,2,2}}, {0.75, {2,2,0}}} 
    });
    // 6
    funcs.push_back({ {{4.0*std::sqrt(5.0/8.0), {0,1,3}}, {-3.0*std::sqrt(5.0/8.0), {2,1,1}}, {-3.0*std::sqrt(5.0/8.0), {0,3,1}}} });
    // 7
    funcs.push_back({ 
        {{1.5*std::sqrt(5.0), {2,0,2}}, {-0.25*std::sqrt(5.0), {4,0,0}}, 
         {-1.5*std::sqrt(5.0), {0,2,2}}, {0.25*std::sqrt(5.0), {0,4,0}}} 
    });
    // 8
    funcs.push_back({ {{std::sqrt(35.0/8.0), {3,0,1}}, {-3.0*std::sqrt(35.0/8.0), {1,2,1}}} });
    // 9
    funcs.push_back({ 
        {{std::sqrt(35.0)/8.0, {4,0,0}}, {std::sqrt(35.0)/8.0, {0,4,0}}, 
         {-3.0*std::sqrt(35.0)/4.0, {2,2,0}}} 
    });
    
    return funcs;
}
