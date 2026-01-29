#ifndef ANGLE_GRID_H
#define ANGLE_GRID_H

#include <vector>
#include <string>
#include <iostream>
#include "rotation.h"

// AngleGrid storage
// Stores a list of orientations (alpha, beta, gamma)
// and associated quadrature weights.

struct Orientation {
    double alpha;
    double beta;
    double gamma;
    double weight;
};

class AngleGrid {
public:
    std::vector<Orientation> points;
    
    AngleGrid();
    
    // Grid Generation Methods (Clears existing points)
    void GenerateHardcoded(); // Full 150 points from reference
    void GenerateGeometric(int n_alpha, int n_beta); // Uniform grid
    void GenerateRepulsion(int n_points, int seed=1234); // Fibonacci + Repulsion
    
    // Modifiers
    void ApplyGammaSampling(int n_gamma); // Expands current grid with gamma values [0, pi]
    
    // I/O
    void LoadFromFile(const std::string& filename); 
    void SaveToFile(const std::string& filename) const;
    
    // Access
    size_t Size() const { return points.size(); }
    const Orientation& Get(size_t i) const { return points[i]; }
    
    // Helpers
    void Clear() { points.clear(); }
};

#endif
