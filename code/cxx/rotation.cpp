#include "rotation.h"
#include <cstring>

RotationMatrix::RotationMatrix() {
    SetIdentity();
}

void RotationMatrix::SetIdentity() {
    mat = {1, 0, 0,
           0, 1, 0,
           0, 0, 1};
}

// Z-Y-Z Euler Angles
// R = R_z(alpha) * R_y(beta) * R_z(gamma)
// Note: The reference code RotnMatr::EulerRotnMatr implementation:
// rotn[0] = cosC*cosA-cosB*sinA*sinC; ...
// Let's verify this matches Z-Y-Z convention explicitly.
// R_z(a) = [[ca -sa 0], [sa ca 0], [0 0 1]]
// R_y(b) = [[cb 0 sb], [0 1 0], [-sb 0 cb]]
// R_z(g) = [[cg -sg 0], [sg cg 0], [0 0 1]]
//
// Reference implementation in rotnmatr.C:
// rotn[0] = cosC*cosA-cosB*sinA*sinC
// rotn[1] = cosC*sinA+cosB*cosA*sinC
// rotn[2] = sinC*sinB
// This looks like active rotation or passive? 
// Standard Goldstein Z-Y-Z matrix (Passive / Coordinate Transform):
// x' = R x
// R11 = cos(psi)cos(phi) - cos(theta)sin(phi)sin(psi)  (where phi=alpha, theta=beta, psi=gamma)
// My Reference Code uses:
// alpha (A), beta (B), gamma (C).
// rotn[0] = cosC*cosA - cosB*sinA*sinC. Matches Goldstein R11.
// So the reference implements standard Z-Y-Z Rotation Matrix.
void RotationMatrix::SetFromEuler(double alpha, double beta, double gamma) {
    // Standard Z-Y-Z Euler Angles
    // R = R_z(alpha) * R_y(beta) * R_z(gamma)
    
    double ca = std::cos(alpha);
    double sa = std::sin(alpha);
    double cb = std::cos(beta);
    double sb = std::sin(beta);
    double cg = std::cos(gamma);
    double sg = std::sin(gamma);

    // R11 = ca*cb*cg - sa*sg
    mat[0] = ca*cb*cg - sa*sg;
    // R12 = -ca*cb*sg - sa*cg
    mat[1] = -ca*cb*sg - sa*cg;
    // R13 = ca*sb
    mat[2] = ca*sb;

    // R21 = sa*cb*cg + ca*sg
    mat[3] = sa*cb*cg + ca*sg;
    // R22 = -sa*cb*sg + ca*cg
    mat[4] = -sa*cb*sg + ca*cg;
    // R23 = sa*sb
    mat[5] = sa*sb;

    // R31 = -sb*cg
    mat[6] = -sb*cg;
    // R32 = sb*sg
    mat[7] = sb*sg;
    // R33 = cb
    mat[8] = cb;
}

void RotationMatrix::SetFromAxisAngle(char axis, double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    
    if (axis == 'x' || axis == 'X') {
        mat = {1, 0, 0,
               0, c, s,
               0, -s, c};  // Active? Reference XRotnMatr: 0, c, s, 0, -s, c. Matches.
    } else if (axis == 'y' || axis == 'Y') {
        mat = {c, 0, s,
               0, 1, 0,
               -s, 0, c}; // Reference YRotnMatr matches.
    } else if (axis == 'z' || axis == 'Z') {
        mat = {c, s, 0,
               -s, c, 0,
               0, 0, 1}; // Reference ZRotnMatr matches.
    }
}

double RotationMatrix::Determinant() const {
    return mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]) -
           mat[1] * (mat[3] * mat[8] - mat[5] * mat[6]) +
           mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
}

RotationMatrix RotationMatrix::Transpose() const {
    RotationMatrix t;
    t.mat[0] = mat[0]; t.mat[1] = mat[3]; t.mat[2] = mat[6];
    t.mat[3] = mat[1]; t.mat[4] = mat[4]; t.mat[5] = mat[7];
    t.mat[6] = mat[2]; t.mat[7] = mat[5]; t.mat[8] = mat[8];
    return t;
}

RotationMatrix RotationMatrix::Inverse() const {
    // For rotation matrices, Inverse == Transpose
    return Transpose();
}

std::array<double, 3> RotationMatrix::Apply(const std::array<double, 3>& v) const {
    return {
        mat[0]*v[0] + mat[1]*v[1] + mat[2]*v[2],
        mat[3]*v[0] + mat[4]*v[1] + mat[5]*v[2],
        mat[6]*v[0] + mat[7]*v[1] + mat[8]*v[2]
    };
}

void RotationMatrix::Apply(double& x, double& y, double& z) const {
    double nx = mat[0]*x + mat[1]*y + mat[2]*z;
    double ny = mat[3]*x + mat[4]*y + mat[5]*z;
    double nz = mat[6]*x + mat[7]*y + mat[8]*z;
    x = nx;
    y = ny;
    z = nz;
}
