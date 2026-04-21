/*
    Photodetachment Suite

    Licensing and provenance notice:
    This file contains logic derived from ezDyson numerical averaging routines.
    Upstream reference: https://iopenshell.usc.edu/downloads/ezdyson/
    See project NOTICE for attribution details and citations.
*/

#ifndef NUM_EIKR_H
#define NUM_EIKR_H

#include <vector>
#include <complex>
#include "dyson.h"
#include "angle_grid.h"
#include "grid.h"

// Class for computing cross sections with numerical REPULSION averaging.
// Provenance: implementation follows ezDyson eikr workflow and conventions.
class NumEikr {
public:
    NumEikr();
    ~NumEikr();

    // Main driver
    void compute(const UniformGrid& labgrid, 
                 const Dyson& dysonL, const Dyson& dysonR,
                 const AngleGrid& anggrid, 
                 const std::vector<double>& energies);

    // Getters for results
    double get_sigma_par(int k_idx) const { return cpar[k_idx]; }
    double get_sigma_perp(int k_idx) const { return cperp[k_idx]; }

private:
    std::vector<double> cpar;  // Parallel cross section
    std::vector<double> cperp; // Perpendicular cross section
    
    // Internal method for calculation
    void calc_eikr_sq(const UniformGrid& labgrid, 
                      const Dyson& dysonL, const Dyson& dysonR,
                      const AngleGrid& anggrid, 
                      const std::vector<double>& k_values);
};

#endif // NUM_EIKR_H
