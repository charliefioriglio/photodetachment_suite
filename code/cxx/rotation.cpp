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
void RotationMatrix::SetFromEuler(double alpha, double beta, double gamma) {
    
    double ca = std::cos(alpha);
    double sa = std::sin(alpha);
    double cb = std::cos(beta);
    double sb = std::sin(beta);
    double cg = std::cos(gamma);
    double sg = std::sin(gamma);

    mat[0] = ca*cb*cg - sa*sg;
    mat[1] = -ca*cb*sg - sa*cg;
    mat[2] = ca*sb;

    mat[3] = sa*cb*cg + ca*sg;
    mat[4] = -sa*cb*sg + ca*cg;
    mat[5] = sa*sb;

    mat[6] = -sb*cg;
    mat[7] = sb*sg;
    mat[8] = cb;
}

void RotationMatrix::SetFromAxisAngle(char axis, double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    
    if (axis == 'x' || axis == 'X') {
        mat = {1, 0, 0,
               0, c, s,
               0, -s, c};
    } else if (axis == 'y' || axis == 'Y') {
        mat = {c, 0, s,
               0, 1, 0,
               -s, 0, c};
    } else if (axis == 'z' || axis == 'Z') {
        mat = {c, s, 0,
               -s, c, 0,
               0, 0, 1};
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
