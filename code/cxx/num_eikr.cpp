#include "num_eikr.h"
#include <cmath>
#include <iostream>
#include <cstring>
#include <array>

// Internal helper for ezDyson-compatible Rotation Logic
struct EzRotation {
    double mat[9];

    // Calc rotn which is Transpose of Active ZXZ Matrix
    // Matches ezDyson/rotnmatr.C: EulerRotnMatr(alpha, beta, gamma)
    void set_euler_zxz_transpose(double alpha, double beta, double gamma) {
        double cosA = std::cos(alpha), sinA = std::sin(alpha);
        double cosB = std::cos(beta), sinB = std::sin(beta);
        double cosC = std::cos(gamma), sinC = std::sin(gamma);

        // Row 0
        mat[0] = cosC*cosA - cosB*sinA*sinC; // Matches Active ZXZ R_00 ? No, R_00 = cAcG - cBsAsG. Same.
        mat[1] = cosC*sinA + cosB*cosA*sinC; // R_10? sAcG + cBcAsG. Matches.
        mat[2] = sinC*sinB;                  // R_20? sBsG. Matches.

        // Row 1
        mat[3] = (-sinC)*cosA - cosB*sinA*cosC; // R_01? -cA sG - cB sA cG. Matches R_01 of Active ZXZ?
                                                // R_01 = cA(-sG) - sA(cB cG) = -cAsG - cBsAcG. Matches.
        mat[4] = (-sinC)*sinA + cosB*cosA*cosC; // R_11? sA(-sG) + cA(cB cG). Matches.
        mat[5] = cosC*sinB;                     // R_21? sB cG. Matches.

        // Row 2
        mat[6] = sinB*sinA;      // R_02? sA sB. Matches.
        mat[7] = (-sinB)*cosA;   // R_12? -cA sB. Matches.
        mat[8] = cosB;           // R_22? cB. Matches.
        
        // Note: ezDyson implementation seems to store R^T relative to Active ZXZ?
        // Wait, if mat[1] (0,1) matches R_10, then it IS Transpose.
        // My check above: mat[1] matches R_10. 
        // So this matrix IS R^T.
        // And it transforms Lab -> Mol.
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
    // E = k^2 / 2 => k = sqrt(2*E)
    // Careful with units: Input energies are in eV.
    // k (a.u.) = sqrt(2 * E_eV / 27.211386)
    // From tools.h: HAR_TO_EV = 27.211386
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
    // Assuming labgrid describes the integration volume in Lab Frame
    // Just iterating points x,y,z
    // We assume uniform steps
    double dx = labgrid.dx;
    double dy = labgrid.dy;
    double dz = labgrid.dz;
    double dV = dx*dy*dz;
    double dV2 = dV * dV; // Precompute dV^2 for |Integral|^2?
    // Wait. ezDyson computes Integral(Psi * ...)^2.
    // Integral ~ Sum(val * dV).
    // |Integral|^2 ~ |Sum|^2.
    // ezDyson code: tmp = (SumL * SumR).Re().
    // Inside loop: Lcklm += val * dV? No.
    // In eikr.C loop: Lcklm += totalLDys_par (which is val).
    // Then after loop: tmp_par = (SumL * SumR).Re() * dVV * tmpavg.
    // dVV = dV * dV. This matches |Integral*dV|^2.
    
    double x0 = labgrid.xmin;
    double y0 = labgrid.ymin;
    double z0 = labgrid.zmin;
    int nx = labgrid.nx;
    int ny = labgrid.ny;
    int nz = labgrid.nz;
    
    // Thread-local accumulation (since we parallelize over orientations? Or just simple loop)
    // We will loop over orientations and Inside loop over energies.
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

        #pragma omp for
        for (int v = 0; v < n_orientations; ++v) {
            const auto& p = anggrid.points[v];
            
            // Set rotation for this thread's orientation
            rot_local.set_euler_zxz_transpose(p.alpha, p.beta, p.gamma);
            double weight = p.weight;
            
            // Pre-allocate sums for all energies for this orientation
            // Reset for each orientation
            std::vector<std::complex<double>> sumL_par(n_energies, 0.0);
            std::vector<std::complex<double>> sumR_par(n_energies, 0.0);
            std::vector<std::complex<double>> sumL_x(n_energies, 0.0);
            std::vector<std::complex<double>> sumR_x(n_energies, 0.0);
            std::vector<std::complex<double>> sumL_y(n_energies, 0.0);
            std::vector<std::complex<double>> sumR_y(n_energies, 0.0);
            
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
