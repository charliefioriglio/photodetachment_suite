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
             // Fallback or Error? Ideally we should implement general recursion if needed, 
             // but for now we rely on the table.
             return 0.0; 
        }

        // int L_idx = -1; // Index in cgc[j1][?]
        // Table uses: cgc[l1][L_idx][m1+l1][m2+1]
        // L_idx mapping from beta.cpp: L_idx = j3 - abs(j1-1); ??
        // In ClebschGordan.cpp comments: "l2=1 always... [l1][L][m1][m2]"
        // Checking beta.cpp usage:
        // double cgc1 = cg.cgc[l][L_idx][m1+l][1]; where L_idx = ltot - std::abs(l-1)
        // ltot is the coupled J (Result J). Here it is j3.
        // So L_idx = j3 - std::abs(j1 - 1);
        
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
