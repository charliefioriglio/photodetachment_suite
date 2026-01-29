#ifndef GAUSS_H
#define GAUSS_H

#include <vector>
#include <cmath>
#include "tools.h"

// Represents a single primitive Cartesian Gaussian
struct CartesianPrimitive {
    double alpha;
    int lx, ly, lz;
    double normalization_constant;
    double coef; // Contraction coefficient

    CartesianPrimitive(double a, int x, int y, int z, double c);

    double evaluate(double x, double y, double z, double cx, double cy, double cz) const;
    static double calculate_normalization(double alpha, int lx, int ly, int lz);
};

// Represents a bundle of primitives (contracted Gaussian)
class ContractedGaussian {
public:
    int lx, ly, lz;
    std::vector<CartesianPrimitive> primitives;

    ContractedGaussian(int x, int y, int z) : lx(x), ly(y), lz(z) {}
    
    void add_primitive(double alpha, double coef);
    double evaluate(double x, double y, double z, double cx, double cy, double cz) const;
    void normalize(); // Adjust coefficients so self-overlap is 1
};

#endif // GAUSS_H
