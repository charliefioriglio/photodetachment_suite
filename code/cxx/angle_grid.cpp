#include "angle_grid.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>

AngleGrid::AngleGrid() {
}

void AngleGrid::GenerateRepulsion(int n_points, int seed) {
    Clear();
    // 1. Fibonacci Sphere Initialization
    struct Vec3 { double x, y, z; };
    std::vector<Vec3> pts(n_points);
    
    double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;
    
    for(int i=0; i<n_points; ++i) {
        double z = 1.0 - 2.0 * (i + 0.5) / n_points; // 1 to -1
        double theta = std::acos(z); // Beta
        double phi = 2.0 * M_PI * i / golden_ratio; // Alpha
        phi = std::fmod(phi, 2.0 * M_PI);
        
        pts[i] = {
            std::sin(theta) * std::cos(phi),
            std::sin(theta) * std::sin(phi),
            z
        };
    }
    
    // 2. Perturbation
    std::mt19937 gen(seed);
    std::normal_distribution<> d(0, 1);
    double perturb_strength = 0.1;
    
    for(auto& p : pts) {
        Vec3 rand = {d(gen), d(gen), d(gen)};
        // Project onto tangent plane: rand = rand - (rand . p) * p
        double dot = rand.x*p.x + rand.y*p.y + rand.z*p.z;
        rand.x -= dot*p.x; rand.y -= dot*p.y; rand.z -= dot*p.z;
        // Normalize rand
        double nr = std::sqrt(rand.x*rand.x + rand.y*rand.y + rand.z*rand.z);
        if(nr > 1e-9) {
            rand.x/=nr; rand.y/=nr; rand.z/=nr;
            // Add
            p.x += perturb_strength * rand.x;
            p.y += perturb_strength * rand.y;
            p.z += perturb_strength * rand.z;
            // Renormalize p
            double np = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
            p.x/=np; p.y/=np; p.z/=np;
        }
    }
    
    // 3. Coulomb Repulsion Optimization
    int max_iter = 5000;
    double C = 1e-3;
    double tol = 1e-4;
    
    for(int iter=0; iter<max_iter; ++iter) {
        std::vector<Vec3> disp(n_points, {0,0,0});
        double max_disp = 0;
        
        for(int i=0; i<n_points; ++i) {
            for(int j=i+1; j<n_points; ++j) {
                 double dot = pts[i].x*pts[j].x + pts[i].y*pts[j].y + pts[i].z*pts[j].z;
                 if(dot > 1.0) dot = 1.0; if(dot < -1.0) dot = -1.0;
                 double theta = std::acos(dot);
                 if(theta < 1e-6) continue;
                 
                 // Force direction logic:
                 // Use simple repulsion: F = C/theta^2 * direction(pj -> pi)?
                 // Reference uses cross products to get tangent direction on sphere.
                 // V = cross(vj, vi) -> vector perpendicular to great circle plane
                 // dP_i = cross(V, vi) -> tangent at vi pointing away from vj
                 
                 Vec3 V = {
                     pts[j].y*pts[i].z - pts[j].z*pts[i].y,
                     pts[j].z*pts[i].x - pts[j].x*pts[i].z,
                     pts[j].x*pts[i].y - pts[j].y*pts[i].x
                 };
                 double nV = std::sqrt(V.x*V.x + V.y*V.y + V.z*V.z);
                 if(nV < 1e-9) continue;
                 
                 Vec3 dP_i = {
                     V.y*pts[i].z - V.z*pts[i].y,
                     V.z*pts[i].x - V.x*pts[i].z,
                     V.x*pts[i].y - V.y*pts[i].x
                 };
                 // Normalize tangent
                 double ndPi = std::sqrt(dP_i.x*dP_i.x + dP_i.y*dP_i.y + dP_i.z*dP_i.z);
                 dP_i.x/=ndPi; dP_i.y/=ndPi; dP_i.z/=ndPi;
                 
                 Vec3 dP_j = {
                     pts[j].y*V.z - pts[j].z*V.y,
                     pts[j].z*V.x - pts[j].x*V.z,
                     pts[j].x*V.y - pts[j].y*V.x
                 };
                  double ndPj = std::sqrt(dP_j.x*dP_j.x + dP_j.y*dP_j.y + dP_j.z*dP_j.z);
                 dP_j.x/=ndPj; dP_j.y/=ndPj; dP_j.z/=ndPj;
                 
                 double F = C / (theta * theta);
                 
                 disp[i].x += F * dP_i.x; disp[i].y += F * dP_i.y; disp[i].z += F * dP_i.z;
                 disp[j].x += F * dP_j.x; disp[j].y += F * dP_j.y; disp[j].z += F * dP_j.z;
            }
        }
        
        for(int i=0; i<n_points; ++i) {
            pts[i].x += disp[i].x; pts[i].y += disp[i].y; pts[i].z += disp[i].z;
            double n = std::sqrt(pts[i].x*pts[i].x + pts[i].y*pts[i].y + pts[i].z*pts[i].z);
            pts[i].x /= n; pts[i].y /= n; pts[i].z /= n;
            
            double mag = std::sqrt(disp[i].x*disp[i].x + disp[i].y*disp[i].y + disp[i].z*disp[i].z);
            if(mag > max_disp) max_disp = mag;
        }
        
        if(max_disp < tol) break;
    }
    
    // Convert to Euler ZYZ
    double weight = 1.0 / n_points;
    for(const auto& p : pts) {
        double alpha = std::atan2(p.y, p.x) + M_PI; // [0, 2pi]
        double beta = std::acos(p.z);
        // Default Gamma 0
        points.push_back({alpha, beta, 0.0, weight});
    }
}

void AngleGrid::LoadFromFile(const std::string& filename) {
    Clear();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open angle grid file " << filename << std::endl;
        return;
    }
    double a, b, g, w;
    while (file >> a >> b >> g >> w) {
        points.push_back({a, b, g, w});
    }
}

void AngleGrid::SaveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    file.precision(6);
    file << std::fixed;
    for(const auto& p : points) {
        file << p.alpha << " " << p.beta << " " << p.gamma << " " << p.weight << "\n";
    }
}
