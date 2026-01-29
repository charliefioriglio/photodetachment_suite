#ifndef SHELL_H
#define SHELL_H

#include <vector>
#include <map>
#include "gauss.h"

// Types of pure functions
struct PureFunction {
    std::vector<std::pair<double, std::vector<int>>> components; // (coeff, [lx, ly, lz])
};

class Shell {
public:
    int l;
    bool is_pure;
    std::vector<ContractedGaussian> contracted_functions; // The resulting basis functions (u)

    // Raw input data
    std::vector<double> exponents;
    std::vector<double> coefficients;

    Shell(int angular_mom, bool pure, const std::vector<double>& exps, const std::vector<double>& coeffs);

    // Build the primitives into contracted_functions based on l and is_pure
    void build();

private:
    std::vector<std::vector<int>> get_cartesian_monomials(int l);
    std::vector<PureFunction> get_pure_functions(int l);
    
    // G-shell helpers
    std::vector<PureFunction> get_pure_d();
    std::vector<PureFunction> get_pure_f();
    std::vector<PureFunction> get_pure_g();
};

#endif // SHELL_H
