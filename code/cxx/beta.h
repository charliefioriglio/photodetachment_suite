#ifndef BETA_H
#define BETA_H

#include <vector>
#include "dyson.h"
#include "grid.h"
#include "angle_grid.h"

struct BetaResult {
    double energy;
    double sigma_par;
    double sigma_perp;
    double beta;
};

class BetaCalculator {
public:
    static std::vector<BetaResult> CalculateBeta(
        const Dyson& dyson_L,
        const Dyson& dyson_R,
        const UniformGrid& grid,
        const AngleGrid& angle_grid,
        const std::vector<double>& photoelectron_energies_ev
    );

    static std::vector<BetaResult> CalculateBetaAnalytic(
        const Dyson& dyson_L,
        const Dyson& dyson_R,
        const UniformGrid& grid,
        const std::vector<double>& photoelectron_energies_ev,
        int l_max = 3
    );

    static std::vector<BetaResult> CalculateBetaPointDipole(
        const Dyson& dyson_L,
        const Dyson& dyson_R,
        const UniformGrid& grid,
        const std::vector<double>& photoelectron_energies_ev,
        double dipole_strength,
        int l_max = 3
    );

private:
    static std::complex<double> ComputeNumericalMatrixElement(
        const Dyson& dyson,
        const UniformGrid& grid,
        const double* k_vec,
        const double* pol_vec
    );
};

#endif
