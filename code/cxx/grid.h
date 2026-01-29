#ifndef GRID_H
#define GRID_H

#include <vector>
#include <string>

class UniformGrid {
public:
    double xmin, xmax, ymin, ymax, zmin, zmax;
    int nx, ny, nz;
    
    // Step sizes
    double dx, dy, dz;

    // Derived
    std::vector<double> x_vals, y_vals, z_vals;

    UniformGrid(double x0, double x1, double y0, double y1, double z0, double z1, double step);
    
    void write_cube_file(const std::string& filename, /* const Dyson& dyson */ const std::vector<double>& data, int n_atoms /* and atom data */);
    
    // Simple binary writer for Python integration efficiency
    // Format: NX NY NZ X0 Y0 Z0 DX DY DZ [DATA...]
    void write_binary(const std::string& filename, const std::vector<double>& data);
};

#endif // GRID_H
