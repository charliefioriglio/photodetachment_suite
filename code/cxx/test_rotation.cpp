#include "rotation.h"
#include "angle_grid.h"
#include <iostream>
#include <cmath>
#include <cassert>

bool approx_equal(double a, double b, double tol=1e-6) {
    return std::abs(a - b) < tol;
}

void TestRotation() {
    std::cout << "Testing Rotation Matrix..." << std::endl;
    RotationMatrix R;
    
    // 1. Identity
    double x = 1, y = 2, z = 3;
    R.Apply(x, y, z);
    assert(approx_equal(x, 1) && approx_equal(y, 2) && approx_equal(z, 3));
    
    // 2. Rotation around Z by 90 degrees (Active rotation of OBJECT)
    // Point (1,0,0) -> (0,1,0)
    // Matrix: [[0 -1 0], [1 0 0], [0 0 1]] ?
    // In code: ZRotnMatr: [[c s 0], [-s c 0], [0 0 1]].
    // My code `SetFromAxisAngle` for 'z': [[c s 0], [-s c 0], [0 0 1]].
    // This is PASSIVE rotation (Coordinate Transform).
    // x' = x cos t + y sin t
    // y' = -x sin t + y cos t
    // If we rotate AXES by 90 deg z (CCW):
    // Old X axis becomes New Y axis.
    // Point P=(1,0,0)_old matches P=(0,-1,0)_new ???
    // Wait.
    // Let's check logic.
    // Rz(90): c=0, s=1.
    // x' = 0 + y*1 = y
    // y' = -x + 0 = -x
    // z' = z
    // Point (1,0,0) -> (0,-1,0). Correct for coordinate transform.
    
    R.SetFromAxisAngle('z', M_PI / 2.0);
    x = 1; y = 0; z = 0;
    R.Apply(x, y, z);
    
    // Check against (0, -1, 0)
    std::cout << "Rot(Z, 90) * (1,0,0) = (" << x << ", " << y << ", " << z << ")" << std::endl;
    assert(approx_equal(x, 0) && approx_equal(y, -1) && approx_equal(z, 0));
    std::cout << "  Passed." << std::endl;
    
    // 3. Euler Angles Z-Y-Z
    // (alpha=90, beta=90, gamma=0)
    // R = Rz(0) * Ry(90) * Rz(90) ??? No, my code: Rz(alpha)*Ry(beta)*Rz(gamma).
    // Order matters. 
    // Passive: M = R_gamma * R_beta * R_alpha ??
    // My code:
    // row 0: cc*ca - cb*sa*sc (Matches Goldstein)
    // Goldstein order convention: First alpha about z, then beta about y', then gamma about z''.
    // Matrix M corresponds to v_body = M * v_space.
    // Let's test.
    // alpha=90 (z), beta=90 (new y), gamma=0.
    // 1. Rotate Z by 90: X->Y, Y->-X
    // 2. Rotate Y' (old -X) by 90: new axes.
    // Let's trust the formula validation: M * v_lab = v_mol.
    // If molecule starts aligned with lab.
    // We rotate lab axes to get molecular axes? Or rotate molecule?
    // Usually "Euler Angles" descriptor implies orientation of rigid body.
    // v_body = M v_spatial.
    
    R.SetFromEuler(M_PI/2, M_PI/2, 0);
    // Formula:
    // a=90, b=90, g=0.
    // ca=0, sa=1. cb=0, sb=1. cc=1, sc=0.
    // row0: 1*0 - 0 = 0.
    // row0[1]: 1*1 + 0 = 1.
    // row0[2]: 0.
    // row1: -0 - 0 = 0.
    // row1[1]: -0 + 0 = 0.
    // row1[2]: 1.
    // row2: 1*1 = 1.
    // row2[1]: -0 = 0.
    // row2[2]: 0.
    // Matrix:
    // 0 1 0
    // 0 0 1
    // 1 0 0
    //
    // Apply to (1,0,0) (Lab X axis):
    // x' = 0*1 + 1*0 + 0 = 0
    // y' = 0*1 + 0 + 1*0 = 0
    // z' = 1*1 = 1
    // Result (0,0,1).
    // So Frame X -> Frame Z.
    // Frame Y -> (1,0,0) -> Frame X.
    // Frame Z -> (0,1,0) -> Frame Y.
    // X->Z, Y->X, Z->Y. Cyclic permutation.
    // Correct.
    
    x = 1; y = 0; z = 0;
    R.Apply(x, y, z);
    std::cout << "Euler(90,90,0) * (1,0,0) = (" << x << ", " << y << ", " << z << ")" << std::endl;
    assert(approx_equal(x, 0) && approx_equal(y, 0) && approx_equal(z, 1));
    std::cout << "  Passed." << std::endl;
}

void TestGrid() {
    std::cout << "Testing Angle Grid..." << std::endl;
    AngleGrid grid;
    
    // 1. Hardcoded
    grid.GenerateHardcoded();
    std::cout << "Hardcoded Points: " << grid.Size() << std::endl;
    assert(grid.Size() == 150);
    // Verify point 0
    assert(approx_equal(grid.Get(0).alpha, 1.07393));
    grid.SaveToFile("grid_hardcoded.txt");
    std::cout << "  Passed (Saved to grid_hardcoded.txt)." << std::endl;
    
    // 2. Geometric
    grid.GenerateGeometric(10, 5); // 10 alpha, 5 beta = 50 points
    std::cout << "Geometric Points: " << grid.Size() << std::endl;
    assert(grid.Size() == 50);
    // Check weight sum
    double w_sum = 0;
    for(size_t i=0; i<grid.Size(); ++i) w_sum += grid.Get(i).weight;
    std::cout << "  Weight Sum: " << w_sum << std::endl;
    assert(approx_equal(w_sum, 1.0));
    grid.SaveToFile("grid_geom.txt");
    std::cout << "  Passed (Saved to grid_geom.txt)." << std::endl;
    
    // 3. Repulsion
    std::cout << "Generating Repulsion Grid (50 points)..." << std::endl;
    grid.GenerateRepulsion(50);
    std::cout << "Repulsion Points: " << grid.Size() << std::endl;
    assert(grid.Size() == 50);
    grid.SaveToFile("grid_rep.txt");
    std::cout << "  Passed (Saved to grid_rep.txt)." << std::endl;
    
    // 4. Gamma Sampling
    std::cout << "Applying Gamma Sampling (N=2)..." << std::endl;
    grid.ApplyGammaSampling(2);
    std::cout << "Expanded Points: " << grid.Size() << std::endl;
    assert(grid.Size() == 100); // 50 * 2
    assert(approx_equal(grid.Get(0).gamma, 0.0));
    assert(approx_equal(grid.Get(1).gamma, M_PI/2.0)); // Check step logic
    std::cout << "  Passed." << std::endl;
}

int main() {
    TestRotation();
    TestGrid();
    std::cout << "All Tests Passed." << std::endl;
    return 0;
}
