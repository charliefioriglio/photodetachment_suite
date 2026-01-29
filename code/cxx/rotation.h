#ifndef ROTATION_H
#define ROTATION_H

#include <vector>
#include <array>
#include <cmath>
#include <iostream>

// Standard Z-Y-Z Euler Angle Convention
// Alpha: Rotation around Z
// Beta:  Rotation around new Y'
// Gamma: Rotation around new Z''

class RotationMatrix {
public:
    // Storage for 3x3 matrix (row-major)
    std::array<double, 9> mat;

    RotationMatrix();
    
    // Setters / Generators
    void SetIdentity();
    void SetFromEuler(double alpha, double beta, double gamma); // Z-Y-Z
    void SetFromAxisAngle(char axis, double angle); // 'x', 'y', 'z'

    // Operations
    double Determinant() const;
    RotationMatrix Inverse() const;
    RotationMatrix Transpose() const;

    // Application
    // Rotate a vector v_rot = R * v_orig
    // If R represents rotation of FRAME, then v_new_frame = R * v_old_frame? 
    // Or does it represent rotation of OBJECT?
    // Reference RotnMatr methods: "XLabMol" -> convert x coord from molec to lab frame.
    // x_lab = R[0]*x_mol + R[1]*y_mol + R[2]*z_mol
    // This implies v_lab = R * v_mol. 
    // So R is the transformation M_mol->lab.
    
    std::array<double, 3> Apply(const std::array<double, 3>& v) const;
    void Apply(double& x, double& y, double& z) const;
};

#endif
