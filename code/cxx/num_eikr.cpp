#include "num_eikr.h"
#include <cmath>
#include <iostream>
#include <cstring>
#include <array>

// Internal helper for Rotation Logic
struct EzRotation {
    double mat[9];

    // Calc rotn which is Transpose of Active ZYZ Matrix
    // Matches ezDyson/rotnmatr.C: EulerRotnMatr(alpha, beta, gamma)
    void set_euler_zxz_transpose(double alpha, double beta, double gamma) {
        double cosA = std::cos(alpha), sinA = std::sin(alpha);
        double cosB = std::cos(beta), sinB = std::sin(beta);
        double cosC = std::cos(gamma), sinC = std::sin(gamma);

        // Row 0
        mat[0] = cosC*cosA - cosB*sinA*sinC;
        mat[1] = cosC*sinA + cosB*cosA*sinC;
        mat[2] = sinC*sinB;

        // Row 1
        mat[3] = (-sinC)*cosA - cosB*sinA*cosC;
                                            
        mat[4] = (-sinC)*sinA + cosB*cosA*cosC;
        mat[5] = cosC*sinB;                     

        // Row 2
        mat[6] = sinB*sinA;
        mat[7] = (-sinB)*cosA;
        mat[8] = cosB;
    }

    // Transform Lab -> Mol
    void transform(double xL, double yL, double zL, double& xM, double& yM, double& zM) const {
        xM = mat[0]*xL + mat[1]*yL + mat[2]*zL;
        yM = mat[3]*xL + mat[4]*yL + mat[5]*zL;
        zM = mat[6]*xL + mat[7]*yL + mat[8]*zL;
    }
};

NumEikr::NumEikr() {}
NumEikr::~NumEikr() {}

void NumEikr::compute(const UniformGrid& labgrid, 
                      const Dyson& dysonL, const Dyson& dysonR,
                      const AngleGrid& anggrid, 
                      const std::vector<double>& energies) {
                      
    std::vector<double> k_values;
    k_values.reserve(energies.size());
    // Convert E(eV) to k(a.u.)
    const double HAR_TO_EV = 27.211386;
    for(double e : energies) {
        k_values.push_back(std::sqrt(2.0 * e / HAR_TO_EV));
    }
    
    // Allocate results
    cpar.assign(k_values.size(), 0.0);
    cperp.assign(k_values.size(), 0.0);
    
    calc_eikr_sq(labgrid, dysonL, dysonR, anggrid, k_values);
}

void NumEikr::calc_eikr_sq(const UniformGrid& labgrid, 
                           const Dyson& dysonL, const Dyson& dysonR,
                           const AngleGrid& anggrid, 
                           const std::vector<double>& k_values) {
                           
    int n_energies = k_values.size();
    int n_orientations = anggrid.points.size();
    
    // Grid parameters
    double dx = labgrid.dx;
    double dy = labgrid.dy;
    double dz = labgrid.dz;
    double dV = dx*dy*dz;
    double dV2 = dV * dV;
    double x0 = labgrid.xmin;
    double y0 = labgrid.ymin;
    double z0 = labgrid.zmin;
    int nx = labgrid.nx;
    int ny = labgrid.ny;
    int nz = labgrid.nz;
    
    // Results accumulate to cpar[k], cperp[k].
    
    EzRotation rot;
    
    // Thread-local accumulation
    // Outer parallel region
    #pragma omp parallel
    {
        // Thread-private accumulators
        std::vector<double> cpar_local(n_energies, 0.0);
        std::vector<double> cperp_local(n_energies, 0.0);
        
        // Private rotation object per thread
        EzRotation rot_local;

        // Pre-allocate sums for all energies for this thread
        // These will be reused for each orientation to avoid heap allocations
        std::vector<std::complex<double>> sumL_par(n_energies, 0.0);
        std::vector<std::complex<double>> sumR_par(n_energies, 0.0);
        std::vector<std::complex<double>> sumL_x(n_energies, 0.0);
        std::vector<std::complex<double>> sumR_x(n_energies, 0.0);
        std::vector<std::complex<double>> sumL_y(n_energies, 0.0);
        std::vector<std::complex<double>> sumR_y(n_energies, 0.0);

        #pragma omp for
        for (int v = 0; v < n_orientations; ++v) {
            const auto& p = anggrid.points[v];
            
            // Set rotation for this thread's orientation
            // Match ezDyson convention: grid_alpha -> rotation_gamma, rotation_alpha=0
            // This ensures the body-frame polarization vector spans full S²
            rot_local.set_euler_zxz_transpose(0.0, p.beta, p.alpha);
            double weight = p.weight;
            
            // Reset for each orientation
            std::fill(sumL_par.begin(), sumL_par.end(), 0.0);
            std::fill(sumR_par.begin(), sumR_par.end(), 0.0);
            std::fill(sumL_x.begin(), sumL_x.end(), 0.0);
            std::fill(sumR_x.begin(), sumR_x.end(), 0.0);
            std::fill(sumL_y.begin(), sumL_y.end(), 0.0);
            std::fill(sumR_y.begin(), sumR_y.end(), 0.0);
            
            // Loop Spatial Grid
            // Optimization: Iterate flat index if possible, or keep loops
            for(int ix=0; ix<nx; ++ix) {
                double xL = x0 + ix*dx;
                for(int iy=0; iy<ny; ++iy) {
                    double yL = y0 + iy*dy;
                    for(int iz=0; iz<nz; ++iz) {
                        double zL = z0 + iz*dz;
                        
                        // Rotate Coords Lab -> Mol
                        double xM, yM, zM;
                        rot_local.transform(xL, yL, zL, xM, yM, zM);
                        
                        // Evaluate Dyson
                        double valL = dysonL.evaluate(xM, yM, zM);
                        double valR = dysonR.evaluate(xM, yM, zM);
                        
                        // Check threshold to skip negligible points
                         if (std::abs(valL) < 1.0e-15 && std::abs(valR) < 1.0e-15) continue;

                        // Dipole Operator (r . eps) in Lab Frame
                        // ezDyson fixes Polarization along Z (eps = z).
                        // So Operator is always zL.
                        
                        std::complex<double> termL_par = valL * zL;
                        std::complex<double> termR_par = valR * zL;
                        
                        // Perp X: Dipole Z, k || X
                        std::complex<double> termL_x = valL * zL; 
                        std::complex<double> termR_x = valR * zL;
                        
                        // Perp Y: Dipole Z, k || Y
                        std::complex<double> termL_y = valL * zL;
                        std::complex<double> termR_y = valR * zL;
                        
                        // Loop Energies
                        for(int k=0; k<n_energies; ++k) {
                            double kval = k_values[k];
                            // Plane Wave part: exp(i * k * r_lab . k_hat)
                            
                            double kz = kval * zL;
                            double kx = kval * xL;
                            double ky = kval * yL;
                            
                            std::complex<double> exp_kz(std::cos(kz), std::sin(kz));
                            std::complex<double> exp_kx(std::cos(kx), std::sin(kx));
                            std::complex<double> exp_ky(std::cos(ky), std::sin(ky));
                            
                            sumL_par[k] += termL_par * exp_kz;
                            sumR_par[k] += termR_par * std::conj(exp_kz);
                            
                            sumL_x[k] += termL_x * exp_kx;
                            sumR_x[k] += termR_x * std::conj(exp_kx);
                            
                            sumL_y[k] += termL_y * exp_ky;
                            sumR_y[k] += termR_y * std::conj(exp_ky);
                        }
                    }
                }
            } // End Grid
            
            // Accumulate to thread-local cross sections
            for(int k=0; k<n_energies; ++k) {
                double term_par = std::real(sumL_par[k] * sumR_par[k]);
                double term_x = std::real(sumL_x[k] * sumR_x[k]);
                double term_y = std::real(sumL_y[k] * sumR_y[k]);
                
                cpar_local[k]  += term_par * dV2 * weight;
                cperp_local[k] += 0.5 * (term_x + term_y) * dV2 * weight;
            }
            
        } // End Orientations Loop
        
        // Critical Section to merge thread-local results
        #pragma omp critical
        {
            for(int k=0; k<n_energies; ++k) {
                cpar[k] += cpar_local[k];
                cperp[k] += cperp_local[k];
            }
        }
    } // End Parallel Region
}
