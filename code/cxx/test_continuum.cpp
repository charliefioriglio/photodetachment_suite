#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "continuum.h"
#include "rotation.h"
#include "tools.h" // for approx_equal

void TestContinuum() {
    std::cout << "Testing Continuum Functions..." << std::endl;
    
    double k_vec[3] = {0.0, 0.0, 1.0}; // k along Z
    double r_vec[3] = {0.0, 0.5, 0.5}; // r nearby
    
    // 1. Direct Evaluation
    std::complex<double> pw_direct = Continuum::EvaluatePlaneWave(k_vec, r_vec);
    std::cout << "PW Direct: " << pw_direct << std::endl;
    
    // 2. Expansion Convergence
    std::cout << "Checking Expansion Convergence..." << std::endl;
    for(int l=0; l<=10; l+=2) {
        std::complex<double> pw_exp = Continuum::EvaluatePlaneWaveExpansion(k_vec, r_vec, l);
        double diff = std::abs(pw_direct - pw_exp);
        std::cout << "L=" << l << " Diff=" << diff << std::endl;
        if (l >= 6 && diff < 1e-3) {
            std::cout << "  Converged at L=" << l << std::endl;
            break;
        }
    }
    
    // 3. Rotation Invariance check
    std::cout << "Checking Rotation Invariance..." << std::endl;
    RotationMatrix R;
    R.SetFromEuler(M_PI/4, M_PI/4, 0.0); // Arbitrary rotation
    
    // Rotate k and r
    double k_rot[3] = {k_vec[0], k_vec[1], k_vec[2]};
    R.Apply(k_rot[0], k_rot[1], k_rot[2]);
    
    double r_rot[3] = {r_vec[0], r_vec[1], r_vec[2]};
    R.Apply(r_rot[0], r_rot[1], r_rot[2]);
    
    // k . r invariant?
    // Note: If we rotate both vectors by same R, dot product should be invariant.
    std::complex<double> pw_rot = Continuum::EvaluatePlaneWave(k_rot, r_rot);
    
    std::cout << "PW Direct (Ref): " << pw_direct << std::endl;
    std::cout << "PW Rotated:      " << pw_rot << std::endl;
    
    assert(std::abs(pw_direct - pw_rot) < 1e-9);
    std::cout << "  Passed Rotation Invariance." << std::endl;
}

int main() {
    TestContinuum();
    return 0;
}
