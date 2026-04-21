/*
    Photodetachment Suite

    Licensing and provenance notice:
    This file depends on Clebsch-Gordan table data/logic derived from ezDyson.
    Upstream reference: https://iopenshell.usc.edu/downloads/ezdyson/
    See project NOTICE for attribution details and citations.
*/

#ifndef WIGNER_3J_H
#define WIGNER_3J_H

#include "clebsch_gordan.h"
#include <cmath>
#include <iostream>

class Wigner3J {
private:
    ClebschGordan cg;

public:
    Wigner3J() {}

    // Computes Wigner 3j symbol: ( j1  j2  j3 )
    //                            ( m1  m2  m3 )
    // Specifically adapted for the case where j2 = 1 (Dipole coupling) using the internal ClebschGordan table.
    // NOTE: The internal table stores <j1 m1 1 m2 | j3 (m1+m2)>
    // Relation: 
    // ( j1  j2  j3 )  =  (-1)^(j1-j2-m3) * (2*j3+1)^(-1/2) * <j1 m1 j2 m2 | j3 -m3>
    // ( m1  m2  m3 )
    double get_3j_dipole(int j1, int j3, int m1, int m2, int m3) {
        // We only handle j2=1
        int j2 = 1;
        
        // Check selection rules
        if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(m3) > j3) return 0.0;
        if (m1 + m2 + m3 != 0) return 0.0;
        if (std::abs(j1 - j3) > 1) return 0.0;
        
        // Map to Clebsch-Gordan <j1 m1 1 m2 | j3 -m3>
        // Target M = -m3
        // Check if indices are within bounds of the hardcoded table (l <= 10)
        if (j1 > 10 || j3 > 10) {
             return 0.0; 
        }
        
        int idx_j3 = j3 - std::abs(j1 - 1);
        if (idx_j3 < 0 || idx_j3 > 2) return 0.0; // Should be covered by selection rules |j1-j3|<=1
        
        // The table stores <j1 m1 1 m2 | j3 (m1+m2)>
        // We want <j1 m1 1 m2 | j3 -m3>. Note m1+m2 = -m3 is satisfied by 3j condition.
        
        double cg_coeff = cg.cgc[j1][idx_j3][m1+j1][m2+1];
        
        // Conversion factor
        // (-1)^(j1 - j2 - m3) / sqrt(2*j3 + 1)
        // j2 = 1
        int phase_exp = j1 - 1 - m3;
        double phase = (phase_exp % 2 == 0) ? 1.0 : -1.0;
        
        return phase * cg_coeff / std::sqrt(2.0 * j3 + 1.0);
    }
};

#endif // WIGNER_3J_H
