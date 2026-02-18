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

    static std::vector<BetaResult> CalculateBetaPWENumeric(
        const Dyson& dyson_L,
        const Dyson& dyson_R,
        const UniformGrid& grid,
        const std::vector<double>& photoelectron_energies_ev,
        const AngleGrid& angle_grid,
        int l_max = 3
    );

    static std::vector<BetaResult> CalculateBetaPointDipoleNumeric(
        const Dyson& dyson_L,
        const Dyson& dyson_R,
        const UniformGrid& grid,
        const std::vector<double>& photoelectron_energies_ev,
        double dipole_strength,
        const AngleGrid& angle_grid,
        int l_max = 3
    );

    static std::vector<BetaResult> CalculateBetaPhysicalDipole(
        const Dyson& dyson_L,
        const Dyson& dyson_R,
        const UniformGrid& grid,
        const std::vector<double>& photoelectron_energies_ev, // Input is eKE directly for BetaGen?
        // Note: CrossSectionCalculator expects Photon Energies and IE. 
        // BetaGen passes eKE directly. We can pass IE=0 and Energies=eKE.
        double dipole_strength,
        double dipole_length,
        const std::vector<double>& dipole_axis,
        const std::vector<double>& dipole_center,
        const AngleGrid& angle_grid,
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
