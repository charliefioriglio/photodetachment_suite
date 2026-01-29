#include "grid.h"
#include <fstream>
#include <iostream>

UniformGrid::UniformGrid(double x0, double x1, double y0, double y1, double z0, double z1, double step) 
    : xmin(x0), xmax(x1), ymin(y0), ymax(y1), zmin(z0), zmax(z1) {
    
    nx = static_cast<int>((xmax - xmin) / step) + 1;
    ny = static_cast<int>((ymax - ymin) / step) + 1;
    nz = static_cast<int>((zmax - zmin) / step) + 1;
    
    dx = step; dy = step; dz = step;
    
    for(int i=0; i<nx; ++i) x_vals.push_back(xmin + i*dx);
    for(int i=0; i<ny; ++i) y_vals.push_back(ymin + i*dy);
    for(int i=0; i<nz; ++i) z_vals.push_back(zmin + i*dz);
}

void UniformGrid::write_binary(const std::string& filename, const std::vector<double>& data) {
    if (data.size() != static_cast<size_t>(nx * ny * nz)) {
        std::cerr << "Error: Grid data size mismatch" << std::endl;
        return;
    }
    
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
        return;
    }
    
    // Header: NX NY NZ
    out.write(reinterpret_cast<const char*>(&nx), sizeof(int));
    out.write(reinterpret_cast<const char*>(&ny), sizeof(int));
    out.write(reinterpret_cast<const char*>(&nz), sizeof(int));
    
    // Origin: X0 Y0 Z0
    out.write(reinterpret_cast<const char*>(&xmin), sizeof(double));
    out.write(reinterpret_cast<const char*>(&ymin), sizeof(double));
    out.write(reinterpret_cast<const char*>(&zmin), sizeof(double));
    
    // Spacing: DX DY DZ
    out.write(reinterpret_cast<const char*>(&dx), sizeof(double));
    out.write(reinterpret_cast<const char*>(&dy), sizeof(double));
    out.write(reinterpret_cast<const char*>(&dz), sizeof(double));
    
    // Data
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
    out.close();
}
