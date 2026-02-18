#include "beta.h"
#include "cross_section.h"
#include "continuum.h"
#include "rotation.h"
#include "tools.h"
#include "clebsch_gordan.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "math_special.h"
#include <cmath>
#include <complex>
#include <iostream>

std::complex<double> BetaCalculator::ComputeNumericalMatrixElement(
    const Dyson& dyson,
    const UniformGrid& grid,
    const double* k_vec,
    const double* pol_vec
) {
    std::complex<double> integral(0.0, 0.0);
    double dV = grid.dx * grid.dy * grid.dz;
    
    double k_mag = std::sqrt(k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2]);
    int l_max = 10 + int(k_mag * 15.0); 
    
    for (int ix = 0; ix < grid.nx; ++ix) {
        double x = grid.xmin + ix * grid.dx;
        for (int iy = 0; iy < grid.ny; ++iy) {
            double y = grid.ymin + iy * grid.dy;
            for (int iz = 0; iz < grid.nz; ++iz) {
                double z = grid.zmin + iz * grid.dz;
                
                double dyson_val = dyson.evaluate(x, y, z);
                if (std::abs(dyson_val) < 1e-12) continue; 
                
                double r_dot_eps = x * pol_vec[0] + y * pol_vec[1] + z * pol_vec[2];
                double r_vec[3] = {x, y, z};
                std::complex<double> psi_val = Continuum::EvaluatePlaneWaveExpansion(k_vec, r_vec, l_max);
                
                integral += std::conj(psi_val) * r_dot_eps * dyson_val;
            }
        }
    }
    
    return integral * dV;
}

std::vector<BetaResult> BetaCalculator::CalculateBeta(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const AngleGrid& angle_grid,
    const std::vector<double>& photoelectron_energies_ev 
) {
    std::vector<BetaResult> results;
    const double HARTREE_EV = 27.211386;
    
    double pol_lab[3] = {0.0, 0.0, 1.0};      
    double k_par_lab[3] = {0.0, 0.0, 1.0};    
    double k_perp1_lab[3] = {1.0, 0.0, 0.0};  
    double k_perp2_lab[3] = {0.0, 1.0, 0.0};  
    
    for (double E_eV : photoelectron_energies_ev) {
        double E_au = E_eV / HARTREE_EV;
        double k_mag = std::sqrt(2.0 * E_au);
        
        double sum_sigma_par = 0.0;
        double sum_sigma_perp = 0.0;
        
        #pragma omp parallel for reduction(+:sum_sigma_par, sum_sigma_perp)
        for (int i = 0; i < int(angle_grid.points.size()); ++i) {
            const auto& orient = angle_grid.points[i];
            RotationMatrix R;
            R.SetFromEuler(orient.alpha, orient.beta, orient.gamma);
            RotationMatrix RT = R.Transpose();
            
            double pol_mol[3] = {pol_lab[0], pol_lab[1], pol_lab[2]};
            RT.Apply(pol_mol[0], pol_mol[1], pol_mol[2]);
            
            auto rotate_k = [&](const double* k_lab_vec, double k_mag) {
                double k_mol[3] = {k_lab_vec[0]*k_mag, k_lab_vec[1]*k_mag, k_lab_vec[2]*k_mag};
                RT.Apply(k_mol[0], k_mol[1], k_mol[2]);
                return std::vector<double>{k_mol[0], k_mol[1], k_mol[2]};
            };

            auto k_par_mol = rotate_k(k_par_lab, k_mag);
            auto k_perp1_mol = rotate_k(k_perp1_lab, k_mag);
            auto k_perp2_mol = rotate_k(k_perp2_lab, k_mag);
            
            std::complex<double> M_par_L = ComputeNumericalMatrixElement(dyson_L, grid, k_par_mol.data(), pol_mol);
            std::complex<double> M_par_R = ComputeNumericalMatrixElement(dyson_R, grid, k_par_mol.data(), pol_mol);
            
            std::complex<double> A_par_L = M_par_L;
            std::complex<double> A_par_R = std::conj(M_par_R); 
            double sigma_par_orient = std::real(A_par_L * A_par_R);
            
            std::complex<double> M_perp1_L = ComputeNumericalMatrixElement(dyson_L, grid, k_perp1_mol.data(), pol_mol);
            std::complex<double> M_perp1_R = ComputeNumericalMatrixElement(dyson_R, grid, k_perp1_mol.data(), pol_mol);
            double sigma_perp1_orient = std::real(M_perp1_L * std::conj(M_perp1_R));
            
            std::complex<double> M_perp2_L = ComputeNumericalMatrixElement(dyson_L, grid, k_perp2_mol.data(), pol_mol);
            std::complex<double> M_perp2_R = ComputeNumericalMatrixElement(dyson_R, grid, k_perp2_mol.data(), pol_mol);
            double sigma_perp2_orient = std::real(M_perp2_L * std::conj(M_perp2_R));
            
            double sigma_perp_orient = 0.5 * (sigma_perp1_orient + sigma_perp2_orient);
            
            sum_sigma_par += sigma_par_orient * orient.weight;
            sum_sigma_perp += sigma_perp_orient * orient.weight;
        }
        
        double norms = dyson_L.qchem_norm * dyson_R.qchem_norm;
        double sigma_par = sum_sigma_par * norms;
        double sigma_perp = sum_sigma_perp * norms;
        
        double beta = 2.0 * (sigma_par - sigma_perp) / (sigma_par + 2.0 * sigma_perp);
        
        results.push_back({E_eV, sigma_par, sigma_perp, beta});
    }
    
    return results;
}

// Helper to compute Spherical Matrix Elements C_{klm, mu}
// mu = -1, 0, 1 (Spherical Dipole Component)
// result[l*(l+1) + m][mu+1]
std::vector<std::vector<std::complex<double>>> ComputeSphericalMatrixElements(
    const Dyson& dyson,
    const UniformGrid& grid,
    double k,
    int l_max
) {
    int num_lm = (l_max + 1) * (l_max + 1);
    std::vector<std::vector<std::complex<double>>> moments(num_lm, std::vector<std::complex<double>>(3, {0.0, 0.0}));
    
    double dV = grid.dx * grid.dy * grid.dz;
    
    #pragma omp parallel
    {
        auto local_moments = moments; // Thread-local copy
        
        #pragma omp for
        for (int ix = 0; ix < grid.nx; ++ix) {
            double x = grid.xmin + ix * grid.dx;
            for (int iy = 0; iy < grid.ny; ++iy) {
                double y = grid.ymin + iy * grid.dy;
                for (int iz = 0; iz < grid.nz; ++iz) {
                    double z = grid.zmin + iz * grid.dz;
                    
                    double dyson_val = dyson.evaluate(x, y, z);
                    if (std::abs(dyson_val) < 1e-12) continue;
                    
                    double r = std::sqrt(x*x + y*y + z*z);
                    if(r < 1e-10) continue;
                    
                    double theta = std::acos(z/r);
                    double phi = std::atan2(y, x);
                    
                    std::complex<double> Y1_m1 = MathSpecial::SphericalHarmonicY(1, -1, theta, phi);
                    std::complex<double> Y1_0  = MathSpecial::SphericalHarmonicY(1, 0,  theta, phi);
                    std::complex<double> Y1_1  = MathSpecial::SphericalHarmonicY(1, 1,  theta, phi);
                    
                    std::complex<double> dip_m1 = r * std::conj(Y1_m1); 
                    std::complex<double> dip_0  = r * std::conj(Y1_0);
                    std::complex<double> dip_1  = r * std::conj(Y1_1);

                    int idx = 0;
                    for(int l=0; l<=l_max; ++l) {
                        double jl = MathSpecial::SphericalBesselJ(l, k*r);
                        
                        std::complex<double> i_pow_l = (l % 4 == 0) ? 1.0 :
                                                       (l % 4 == 1) ? std::complex<double>(0, 1) :
                                                       (l % 4 == 2) ? -1.0 : std::complex<double>(0, -1);
                                                       
                        double radial_part = jl * dyson_val * dV; 
                        
                        for(int m=-l; m<=l; ++m) {
                            std::complex<double> Ylm = MathSpecial::SphericalHarmonicY(l, m, theta, phi);
                            std::complex<double> Ylm_conj = std::conj(Ylm);
                            
                            std::complex<double> common = radial_part * i_pow_l * Ylm_conj;
                            
                            local_moments[idx][0] += common * dip_m1;
                            local_moments[idx][1] += common * dip_0;
                            local_moments[idx][2] += common * dip_1;
                            
                            idx++;
                        }
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            for(size_t i=0; i<moments.size(); ++i) {
                moments[i][0] += local_moments[i][0];
                moments[i][1] += local_moments[i][1];
                moments[i][2] += local_moments[i][2];
            }
        }
    }
    
    return moments;
}
// Helper to compute Beta from Matrix Elements C_{lm, mu}
// Shared by PWE and Point Dipole
std::vector<BetaResult> ComputeBetaFromMatrixElements(
    const std::vector<std::vector<std::vector<std::complex<double>>>>& matrix_elements_L, // [energy][idx][mu]
    const std::vector<std::vector<std::vector<std::complex<double>>>>& matrix_elements_R,
    const std::vector<double>& energies_ev,
    double qchem_norm_sq,
    int l_max
) {
    std::vector<BetaResult> results;
    ClebschGordan cg; 
    
    // Dipole normalization factor
    // IMPORTANT: Check consistency with PWE derivation.
    // In ComputeSphericalMatrixElements, we didn't apply prefactors.
    double dipole_norm = std::sqrt(4.0 * M_PI / 3.0); 

    for(size_t ie=0; ie<energies_ev.size(); ++ie) {
        double E_eV = energies_ev[ie];
        const auto& C_L = matrix_elements_L[ie];
        const auto& C_R = matrix_elements_R[ie];
        
        // We work with copies to apply norm
        // Or apply norm on the fly
        
        double sigma_par = 0.0;
        double sigma_perp = 0.0;
        
        for (int l=0; l<=l_max; ++l) {
            for (int m1=-l; m1<=l; m1++) {
                
                std::complex<double> Y_par = MathSpecial::SphericalHarmonicY(l, m1, 0.0, 0.0);
                std::complex<double> Y_perp = MathSpecial::SphericalHarmonicY(l, m1, M_PI/2.0, 0.0);
                
                double term_contrib = 0.0;
                
                for (int m21=-l; m21<=l; m21++) {
                   for (int m22=-l; m22<=l; m22++) {
                      for (int v1=0; v1<3; v1++) { 
                         for (int v2=0; v2<3; v2++) {
                           
                            if ((m21+v1) == (m22+v2)) {
                               for (int ltot=(std::abs(l-1)); ltot<=(l+1); ltot++) {
                                  
                                  int idx1 = l*l + (m21+l);
                                  int idx2 = l*l + (m22+l);
                                  
                                  std::complex<double> val1 = C_L[idx1][v1] * dipole_norm;
                                  std::complex<double> val2 = C_R[idx2][v2] * dipole_norm;
                                  
                                  double tmp = std::real(val1 * std::conj(val2)); 
                                  
                                  int L_idx = ltot - std::abs(l-1);
                                  
                                  double cgc1 = cg.cgc[l][L_idx][m1+l][1]; 
                                  double cgc21 = cg.cgc[l][L_idx][m21+l][v1];
                                  double cgc22 = cg.cgc[l][L_idx][m22+l][v2];
                                  
                                  tmp *= cgc1 * cgc1 * cgc21 * cgc22;
                                  tmp /= (2.0 * ltot + 1.0);
                                  
                                  term_contrib += tmp;
                               }
                            }
                         }
                      }
                   }
                }
                
                sigma_par += term_contrib * std::norm(Y_par);
                sigma_perp += term_contrib * std::norm(Y_perp);
            }
        }
        
        // Interference Terms
        for (int l=0; l<=l_max; ++l) {
            for (int m11=-l; m11<=l; m11++) {
                
                std::complex<double> Y1_par = MathSpecial::SphericalHarmonicY(l, m11, 0.0, 0.0);
                std::complex<double> Y1_perp = MathSpecial::SphericalHarmonicY(l, m11, M_PI/2.0, 0.0);
                
                for (int l2=0; l2<=l_max; ++l2) {
                   if (l == l2) continue; 
                   
                   for (int m12=-l2; m12<=l2; m12++) {
                       if (m12 != m11) continue; 
                       
                       std::complex<double> Y2_par = std::conj(MathSpecial::SphericalHarmonicY(l2, m12, 0.0, 0.0));
                       std::complex<double> Y2_perp = std::conj(MathSpecial::SphericalHarmonicY(l2, m12, M_PI/2.0, 0.0));
                       
                       double term_contrib = 0.0;
                       
                       for (int m21=-l; m21<=l; m21++) {
                          for (int m22=-l2; m22<=l2; m22++) {
                             for (int v1=0; v1<3; v1++) {
                                for (int v2=0; v2<3; v2++) {
                                   
                                   if ((m21+v1-1) == (m22+v2-1)) { 
                                      
                                      int idx1 = l*l + (m21+l);
                                      int idx2 = l2*l2 + (m22+l2);
                                      
                                      std::complex<double> val1 = C_L[idx1][v1] * dipole_norm;
                                      std::complex<double> val2 = C_R[idx2][v2] * dipole_norm;
                                      
                                      double tmp = std::real(val1 * std::conj(val2));
                                      
                                      for (int ltot=(std::abs(l-1)); ltot<=(l+1); ltot++) {
                                         for (int ltot2=(std::abs(l2-1)); ltot2<=(l2+1); ltot2++) {
                                            if (ltot == ltot2) {
                                               
                                               int L_idx1 = ltot - std::abs(l-1);
                                               int L_idx2 = ltot2 - std::abs(l2-1);
                                               
                                               double cgc11 = cg.cgc[l][L_idx1][m11+l][1];
                                               double cgc12 = cg.cgc[l2][L_idx2][m12+l2][1];
                                               double cgc21 = cg.cgc[l][L_idx1][m21+l][v1];
                                               double cgc22 = cg.cgc[l2][L_idx2][m22+l2][v2];
                                               
                                               double term = tmp * cgc11 * cgc12 * cgc21 * cgc22;
                                               term /= (2.0 * ltot + 1.0);
                                               
                                               term_contrib += term;
                                            }
                                         }
                                      }
                                   }
                                }
                             }
                          }
                       }
                       sigma_par += term_contrib * std::real(Y1_par * Y2_par); 
                       sigma_perp += term_contrib * std::real(Y1_perp * Y2_perp);
                   }
                }
            }
        }
        
        double prefactor = 3.0 / (4.0 * M_PI);
        sigma_par *= prefactor;
        sigma_perp *= prefactor;
        
        sigma_par *= qchem_norm_sq;
        sigma_perp *= qchem_norm_sq;
        
        double denom = sigma_par + 2.0 * sigma_perp;
        double beta = 0.0;
        if (std::abs(denom) > 1e-18) {
            beta = 2.0 * (sigma_par - sigma_perp) / denom;
        }
        
        results.push_back({E_eV, sigma_par, sigma_perp, beta});
    }
    return results;
}

std::vector<BetaResult> BetaCalculator::CalculateBetaAnalytic(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photoelectron_energies_ev,
    int l_max
) {
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_L;
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_R;
    
    const double HARTREE_EV = 27.211386;
    
    for (double E_eV : photoelectron_energies_ev) {
        double E_au = E_eV / HARTREE_EV;
        double k = std::sqrt(2.0 * E_au);
        
        // For k=0, return zeros
        if (k < 1e-6) {
             int num_lm = (l_max+1)*(l_max+1);
             matrices_L.push_back(std::vector<std::vector<std::complex<double>>>(num_lm, std::vector<std::complex<double>>(3, 0.0)));
             matrices_R.push_back(std::vector<std::vector<std::complex<double>>>(num_lm, std::vector<std::complex<double>>(3, 0.0)));
             continue;
        }

        matrices_L.push_back(ComputeSphericalMatrixElements(dyson_L, grid, k, l_max));
        matrices_R.push_back(ComputeSphericalMatrixElements(dyson_R, grid, k, l_max));
    }
    
    double norm_sq = dyson_L.qchem_norm * dyson_R.qchem_norm;
    return ComputeBetaFromMatrixElements(matrices_L, matrices_R, photoelectron_energies_ev, norm_sq, l_max);
}

// Point Dipole Matrix Elements
#include "point_dipole.h"

// Returns C_{l_in, m, mu}
std::vector<std::vector<std::complex<double>>> ComputePointDipoleMatrixElements(
    const Dyson& dyson,
    const UniformGrid& grid,
    double k,
    double dipole_strength,
    int l_max
) {
    int num_lm = (l_max + 1) * (l_max + 1);
    std::vector<std::vector<std::complex<double>>> moments(num_lm, std::vector<std::complex<double>>(3, {0.0, 0.0}));
    
    PointDipole pd(dipole_strength);
    double dV = grid.dx * grid.dy * grid.dz;
    
    // We iterate over possible incident channels (l_in, m)
    // For Point Dipole, l is coupled, but m is conserved. 
    // We compute the expansion: D_mu(k) = Sum_{l_in, m} Y_{l_in, m}(k) * T_{l_in, m, mu}
    // T_{l_in, m, mu} = < phi | r_mu | Psi_{k, l_in, m} >
    // Psi_{k, l_in, m} = 4 pi * Sum_N [ c_{l_in}^N * i^{L_{eff}} * (u_N/r) * Omega_N(r) ]
    
    // Optimized Strategy:
    // 1. Precompute Overlaps: O_{m, N, mu} = < phi | r_mu | (u_N/r) Omega_N >
    // 2. Assemble T_{l_in, m, mu} = Sum_N (4pi * c_{l_in}^N * i^{L_{eff}}) * O_{m, N, mu}
    
    // Storage for O_{m, N, mu}. 
    // Map key: m (-lmax to lmax). Value: vector of N moments (each N is vec<complex>[3])
    // The number of modes N depends on m.
    std::vector<std::vector<std::vector<std::complex<double>>>> overlaps(2*l_max + 1);
    
    // Initialize overlaps
    for (int lam = -l_max; lam <= l_max; ++lam) {
        Eigensystem sys = pd.GetEigensystem(lam, l_max);
        int n_modes = sys.l_vals.size();
        overlaps[lam + l_max].resize(n_modes, std::vector<std::complex<double>>(3, {0.0, 0.0}));
    }
    
    // Grid Loop
    #pragma omp parallel
    {
        auto local_overlaps = overlaps;
        // Zero out local
        for(auto& v1 : local_overlaps) for(auto& v2 : v1) for(auto& val : v2) val = {0.0, 0.0};
        
        #pragma omp for
        for (int ix = 0; ix < grid.nx; ++ix) {
           for (int iy = 0; iy < grid.ny; ++iy) {
              for (int iz = 0; iz < grid.nz; ++iz) {
                 double x = grid.xmin + ix * grid.dx;
                 double y = grid.ymin + iy * grid.dy;
                 double z = grid.zmin + iz * grid.dz;
                 
                 double dyson_val = dyson.evaluate(x, y, z);
                 if (std::abs(dyson_val) < 1e-12) continue;
                 
                 double r = std::sqrt(x*x + y*y + z*z);
                 if (r < 1e-10) continue;
                 
                 double theta = std::acos(z/r);
                 double phi = std::atan2(y, x);
                 
                 // Precompute dipoles Y1(r)*
                 std::complex<double> Y1_m1 = MathSpecial::SphericalHarmonicY(1, -1, theta, phi);
                 std::complex<double> Y1_0  = MathSpecial::SphericalHarmonicY(1, 0,  theta, phi);
                 std::complex<double> Y1_1  = MathSpecial::SphericalHarmonicY(1, 1,  theta, phi);
                 std::complex<double> dip[3];
                 dip[0] = r * std::conj(Y1_m1);
                 dip[1] = r * std::conj(Y1_0);
                 dip[2] = r * std::conj(Y1_1);
                 
                 for (int lam = -l_max; lam <= l_max; ++lam) {
                      Eigensystem sys = pd.GetEigensystem(lam, l_max);
                      int n_modes = sys.l_vals.size();
                      
                      // Precompute Y_lm(r) for all l in this block
                      std::vector<std::complex<double>> Y_vals(n_modes);
                      for(int i=0; i<n_modes; ++i) {
                          // Match PWE: Use Conjugate of Y_lm(r)
                          Y_vals[i] = std::conj(MathSpecial::SphericalHarmonicY(sys.l_vals[i], lam, theta, phi));
                      }
                      
                      for(int N=0; N<n_modes; ++N) {
                          double eigval = sys.eigvals[N];
                          if (eigval < -0.25) continue; 
                          double L_eff = 0.5 * (-1.0 + std::sqrt(1.0 + 4.0 * eigval));
                          
                          // Radial Part
                          double bessel = MathSpecial::CylBesselJ(L_eff + 0.5, k*r);
                          double radial = std::sqrt(M_PI / (2.0 * k * r)) * bessel; 
                          
                          // Angular Part Omega_N(r)*
                          std::complex<double> omega_r = 0.0;
                          for(int i=0; i<n_modes; ++i) {
                              omega_r += sys.eigvecs[i][N] * Y_vals[i];
                          }
                          
                          // Overlap fragment
                          std::complex<double> fragment = radial * omega_r * dyson_val * dV;
                          
                          // Accumulate
                          for(int mu=0; mu<3; ++mu) {
                              local_overlaps[lam+l_max][N][mu] += fragment * dip[mu];
                          }
                      }
                 }
              }
           }
        }
        
        #pragma omp critical
        {
            for(size_t i=0; i<overlaps.size(); ++i) {
                for(size_t j=0; j<overlaps[i].size(); ++j) {
                    for(int mu=0; mu<3; ++mu) {
                        overlaps[i][j][mu] += local_overlaps[i][j][mu];
                    }
                }
            }
        }
    }
    
    // Assemble Final Matrix Elements T_{l_in, m, mu}
    for (int l_in = 0; l_in <= l_max; ++l_in) {
        for (int m = -l_in; m <= l_in; ++m) {
             // Index in flat vector
             int idx = l_in * l_in + (l_in + m);
             
             // Get Eigensystem for this m
             Eigensystem sys = pd.GetEigensystem(m, l_max);
             int n_modes = sys.l_vals.size();
             
             // Find index of l_in in sys.l_vals
             // sys is built from |lam| to l_max.
             int l_idx_in_sys = -1;
             for(int k=0; k<n_modes; ++k) if(sys.l_vals[k] == l_in) { l_idx_in_sys = k; break; }
             
             if (l_idx_in_sys == -1) continue;
             
             for(int N=0; N<n_modes; ++N) {
                  double eigval = sys.eigvals[N];
                  if(eigval < -0.25) continue;
                  
                  double L_eff = 0.5 * (-1.0 + std::sqrt(1.0 + 4.0 * eigval));
                  
                  // i^L_eff 
                  std::complex<double> phase = std::exp(std::complex<double>(0.0, M_PI * 0.5 * L_eff));
                  
                  // Coefficient: sys.eigvecs[l_idx_in_sys][N]
                  double c_lin = sys.eigvecs[l_idx_in_sys][N];
                  
                  // Match PWE Magnitude: 1.0 (Implicit 4pi skipped)
                  double prefactor = 1.0; 
                  
                  std::complex<double> weight = prefactor * c_lin * phase;
                  
                  for(int mu=0; mu<3; ++mu) {
                       moments[idx][mu] += weight * overlaps[m + l_max][N][mu];
                  }
             }
        }
    } 

    return moments;
}


std::vector<BetaResult> BetaCalculator::CalculateBetaPointDipole(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photoelectron_energies_ev,
    double dipole_strength,
    int l_max
) {
    const double HARTREE_EV = 27.211386;
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_L;
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_R;
    
    for(double E_eV : photoelectron_energies_ev) {
         double E_au = E_eV / HARTREE_EV;
         double k = std::sqrt(2.0 * E_au);
         
         if (k < 1e-6) {
             int num_lm = (l_max+1)*(l_max+1);
             matrices_L.push_back(std::vector<std::vector<std::complex<double>>>(num_lm, std::vector<std::complex<double>>(3, 0.0)));
             matrices_R.push_back(std::vector<std::vector<std::complex<double>>>(num_lm, std::vector<std::complex<double>>(3, 0.0)));
             continue;
         }
         
         matrices_L.push_back(ComputePointDipoleMatrixElements(dyson_L, grid, k, dipole_strength, l_max));
         matrices_R.push_back(ComputePointDipoleMatrixElements(dyson_R, grid, k, dipole_strength, l_max));
    }
    
    double norm_sq = dyson_L.qchem_norm * dyson_R.qchem_norm;
    return ComputeBetaFromMatrixElements(matrices_L, matrices_R, photoelectron_energies_ev, norm_sq, l_max);
}

// Helper for Numeric Averaging (Generic Amplitude Evaluator)
std::vector<BetaResult> BetaCalculator::CalculateBetaPWENumeric(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photoelectron_energies_ev,
    const AngleGrid& angle_grid,
    int l_max
) {
    std::vector<BetaResult> results;
    const double HARTREE_EV = 27.211386;
    
    // Precompute Body-Frame Matrix Elements for all energies
    // Reuse ComputeSphericalMatrixElements
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_L;
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_R;
    
    for (double E_eV : photoelectron_energies_ev) {
        double E_au = E_eV / HARTREE_EV;
        double k = std::sqrt(2.0 * E_au);
        if (k < 1e-6) k = 1e-6; // Avoid zero div
        matrices_L.push_back(ComputeSphericalMatrixElements(dyson_L, grid, k, l_max));
        matrices_R.push_back(ComputeSphericalMatrixElements(dyson_R, grid, k, l_max));
    }
    
    // Lab Frame Vectors
    double pol_lab[3] = {0.0, 0.0, 1.0};      
    double k_par_lab[3] = {0.0, 0.0, 1.0};    
    double k_perp1_lab[3] = {1.0, 0.0, 0.0};  
    double k_perp2_lab[3] = {0.0, 1.0, 0.0};
    
    for(size_t ie=0; ie<photoelectron_energies_ev.size(); ++ie) {
        double E_eV = photoelectron_energies_ev[ie];
        const auto& C_L = matrices_L[ie]; // [lm_idx][mu] (mu=0->-1, 1->0, 2->1)
        const auto& C_R = matrices_R[ie];
        
        double sum_sigma_par = 0.0;
        double sum_sigma_perp = 0.0;
        
        #pragma omp parallel for reduction(+:sum_sigma_par, sum_sigma_perp)
        for(int i=0; i<int(angle_grid.points.size()); ++i) {
            const auto& orient = angle_grid.points[i];
            RotationMatrix R;
            R.SetFromEuler(orient.alpha, orient.beta, orient.gamma);
            RotationMatrix RT = R.Transpose(); // Lab to Body
            
            // Lab Polarization in Body Frame
            double eps_body[3] = {pol_lab[0], pol_lab[1], pol_lab[2]};
            RT.Apply(eps_body[0], eps_body[1], eps_body[2]);
            
            auto ComputeAmp = [&](const double* k_lab, const std::vector<std::vector<std::complex<double>>>& C) -> std::complex<double> {
                double k_body[3] = {k_lab[0], k_lab[1], k_lab[2]};
                RT.Apply(k_body[0], k_body[1], k_body[2]);
                
                double r = std::sqrt(k_body[0]*k_body[0] + k_body[1]*k_body[1] + k_body[2]*k_body[2]);
                if (r < 1e-12) return 0.0;
                double theta = std::acos(k_body[2]/r);
                double phi = std::atan2(k_body[1], k_body[0]);
                
                std::complex<double> total_amp = 0.0;
                
                int idx = 0;
                for(int l=0; l<=l_max; ++l) {
                    for(int m=-l; m<=l; ++m) {
                        std::complex<double> Ylm = MathSpecial::SphericalHarmonicY(l, m, theta, phi);
                        
                        // C coefficients are for spherical components (-1, 0, 1)
                        // eps_body is Cartesian (x, y, z)
                        // Convert eps_body to Spherical:
                        // e_-1 = (x - iy)/sqrt(2)
                        // e_0 = z
                        // e_1 = -(x + iy)/sqrt(2)
                        
                        std::complex<double> e_m1 = (eps_body[0] - std::complex<double>(0,1)*eps_body[1]) / std::sqrt(2.0);
                        std::complex<double> e_0  = eps_body[2];
                        std::complex<double> e_1  = -(eps_body[0] + std::complex<double>(0,1)*eps_body[1]) / std::sqrt(2.0);
                        
                        // C structure from SphericalMatrixElements: [idx][0]=m1, [1]=0, [2]=1
                        std::complex<double> term = C[idx][0] * e_m1 + C[idx][1] * e_0 + C[idx][2] * e_1;
                        
                        total_amp += term * Ylm; 
                        
                        idx++;
                    }
                }
                return total_amp;
            };
            
            std::complex<double> A_par_L = ComputeAmp(k_par_lab, C_L);
            std::complex<double> A_par_R = ComputeAmp(k_par_lab, C_R);
            double sig_par = std::real(A_par_L * std::conj(A_par_R));
            
            std::complex<double> A_perp1_L = ComputeAmp(k_perp1_lab, C_L);
            std::complex<double> A_perp1_R = ComputeAmp(k_perp1_lab, C_R);
            
            std::complex<double> A_perp2_L = ComputeAmp(k_perp2_lab, C_L);
            std::complex<double> A_perp2_R = ComputeAmp(k_perp2_lab, C_R);
            
            double sig_perp = 0.5 * (std::real(A_perp1_L * std::conj(A_perp1_R)) + std::real(A_perp2_L * std::conj(A_perp2_R)));
            
            sum_sigma_par += sig_par * orient.weight;
            sum_sigma_perp += sig_perp * orient.weight;
        }
        
        double norms = dyson_L.qchem_norm * dyson_R.qchem_norm;
        double sigma_par = sum_sigma_par * norms;
        double sigma_perp = sum_sigma_perp * norms;
        
        double denom = sigma_par + 2.0 * sigma_perp;
        double beta = 0.0;
        if (std::abs(denom) > 1e-15) {
            beta = 2.0 * (sigma_par - sigma_perp) / denom;
        }
        
        results.push_back({E_eV, sigma_par, sigma_perp, beta});
    }
    return results;
}

// Helper for Point Dipole Numeric
std::vector<BetaResult> BetaCalculator::CalculateBetaPointDipoleNumeric(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photoelectron_energies_ev,
    double dipole_strength,
    const AngleGrid& angle_grid,
    int l_max
) {
    std::vector<BetaResult> results;
    const double HARTREE_EV = 27.211386;
    PointDipole pd(dipole_strength);

    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_L;
    std::vector<std::vector<std::vector<std::complex<double>>>> matrices_R;
    
    for (double E_eV : photoelectron_energies_ev) {
        double E_au = E_eV / HARTREE_EV;
        double k = std::sqrt(2.0 * E_au);
        if (k < 1e-6) k = 1e-6; 
        matrices_L.push_back(ComputePointDipoleMatrixElements(dyson_L, grid, k, dipole_strength, l_max));
        matrices_R.push_back(ComputePointDipoleMatrixElements(dyson_R, grid, k, dipole_strength, l_max));
    }

    double pol_lab[3] = {0.0, 0.0, 1.0};      
    double k_par_lab[3] = {0.0, 0.0, 1.0};    
    double k_perp1_lab[3] = {1.0, 0.0, 0.0};  
    double k_perp2_lab[3] = {0.0, 1.0, 0.0};
    
    for(size_t ie=0; ie<photoelectron_energies_ev.size(); ++ie) {
        double E_eV = photoelectron_energies_ev[ie];
        const auto& C_L = matrices_L[ie]; 
        const auto& C_R = matrices_R[ie];
        
        double sum_sigma_par = 0.0;
        double sum_sigma_perp = 0.0;
        
        #pragma omp parallel for reduction(+:sum_sigma_par, sum_sigma_perp)
        for(int i=0; i<int(angle_grid.points.size()); ++i) {
            const auto& orient = angle_grid.points[i];
            RotationMatrix R;
            R.SetFromEuler(orient.alpha, orient.beta, orient.gamma);
            RotationMatrix RT = R.Transpose(); 
            
            double eps_body[3] = {pol_lab[0], pol_lab[1], pol_lab[2]};
            RT.Apply(eps_body[0], eps_body[1], eps_body[2]);
            
            // Spherical Pol
            std::complex<double> eps_sph[3];
            eps_sph[0] = (eps_body[0] - std::complex<double>(0,1)*eps_body[1]) / std::sqrt(2.0); // m=-1
            eps_sph[1] = eps_body[2]; // m=0
            eps_sph[2] = -(eps_body[0] + std::complex<double>(0,1)*eps_body[1]) / std::sqrt(2.0); // m=1
            
            auto ComputeAmp = [&](const double* k_lab, const std::vector<std::vector<std::complex<double>>>& C) -> std::complex<double> {
                double k_body[3] = {k_lab[0], k_lab[1], k_lab[2]};
                RT.Apply(k_body[0], k_body[1], k_body[2]);
                
                double r = std::sqrt(k_body[0]*k_body[0] + k_body[1]*k_body[1] + k_body[2]*k_body[2]);
                if (r < 1e-12) return 0.0;
                double theta = std::acos(k_body[2]/r);
                double phi = std::atan2(k_body[1], k_body[0]);
                
                std::complex<double> total_amp = 0.0;
                
                // Point Dipole expansion: Sum_{l_in, m} Y_{l_in, m}(k) * T_{l_in, m, eps}
                // C structure is [l_in * l_in + (l_in+m)][mu+1]
                int idx = 0;
                for(int l_in=0; l_in<=l_max; ++l_in) {
                    for(int m=-l_in; m<=l_in; ++m) {
                        std::complex<double> Ylm = MathSpecial::SphericalHarmonicY(l_in, m, theta, phi);
                        
                        // Dot Product T . eps
                        std::complex<double> T_dot_eps = C[idx][0]*eps_sph[0] + C[idx][1]*eps_sph[1] + C[idx][2]*eps_sph[2];
                        
                        total_amp += Ylm * T_dot_eps;
                        idx++;
                    }
                }
                return total_amp;
            };
            
            std::complex<double> A_par_L = ComputeAmp(k_par_lab, C_L);
            std::complex<double> A_par_R = ComputeAmp(k_par_lab, C_R);
            double sig_par = std::real(A_par_L * std::conj(A_par_R));
            
            std::complex<double> A_perp1_L = ComputeAmp(k_perp1_lab, C_L);
            std::complex<double> A_perp1_R = ComputeAmp(k_perp1_lab, C_R);
            std::complex<double> A_perp2_L = ComputeAmp(k_perp2_lab, C_L);
            std::complex<double> A_perp2_R = ComputeAmp(k_perp2_lab, C_R);
            double sig_perp = 0.5 * (std::real(A_perp1_L * std::conj(A_perp1_R)) + std::real(A_perp2_L * std::conj(A_perp2_R)));
            
            sum_sigma_par += sig_par * orient.weight;
            sum_sigma_perp += sig_perp * orient.weight;
        }
        
        double norms = dyson_L.qchem_norm * dyson_R.qchem_norm;
        double sigma_par = sum_sigma_par * norms;
        double sigma_perp = sum_sigma_perp * norms;
        
        double denom = sigma_par + 2.0 * sigma_perp;
        double beta = 0.0;
        if (std::abs(denom) > 1e-15) {
            beta = 2.0 * (sigma_par - sigma_perp) / denom;
        }
        
        results.push_back({E_eV, sigma_par, sigma_perp, beta});
    }
    return results;
}

#include "physical_dipole.h"
#include <map>

std::vector<BetaResult> BetaCalculator::CalculateBetaPhysicalDipole(
    const Dyson& dyson_L,
    const Dyson& dyson_R,
    const UniformGrid& grid,
    const std::vector<double>& photoelectron_energies_ev,
    double dipole_strength,
    double dipole_length,
    const std::vector<double>& dipole_axis,
    const std::vector<double>& dipole_center,
    const AngleGrid& angle_grid,
    int l_max
) {
    const double HARTREE_EV = 27.211386;
    const double EV_TO_HARTREE = 1.0 / HARTREE_EV;
    std::vector<BetaResult> results;
    
    // 1. Calculate Body-Frame Matrix Elements
    // Pass eKE as Photon Energies with IE=0
    
    // Note: ComputePhysicalDipoleMatrixElements expects vectors of matrix elements per energy
    auto all_matrix_elements = CrossSectionCalculator::ComputePhysicalDipoleMatrixElements(
        dyson_L, dyson_R, grid, photoelectron_energies_ev, 0.0, l_max, 
        dipole_strength, dipole_length, dipole_axis, dipole_center
    );
    
    PhysicalDipole phys(dipole_length, 0.5 * dipole_strength);
    
    double pol_lab[3] = {0.0, 0.0, 1.0};      
    double k_par_lab[3] = {0.0, 0.0, 1.0};    
    double k_perp1_lab[3] = {1.0, 0.0, 0.0};  
    double k_perp2_lab[3] = {0.0, 1.0, 0.0}; 

    for(size_t ie=0; ie<photoelectron_energies_ev.size(); ++ie) {
        double eKE = photoelectron_energies_ev[ie];
        if(eKE <= 0) {
            results.push_back({eKE, 0.0, 0.0, 0.0});
            continue;
        }
        
        // Re-solve angular part to get shapes (Eigenvectors needed for EvaluateAngular)
        std::map<int, PhysicalDipole::Solution> solutions;
        for(int m=-l_max; m<=l_max; ++m) {
            solutions[m] = phys.Solve(eKE * EV_TO_HARTREE, m, l_max);
        }
        
        const auto& elements = all_matrix_elements[ie];
        
        double sum_sigma_par = 0.0;
        double sum_sigma_perp = 0.0;
        
        // Parallelize Orientations
        #pragma omp parallel for reduction(+:sum_sigma_par, sum_sigma_perp)
        for(int i=0; i<int(angle_grid.points.size()); ++i) {
            const auto& orient = angle_grid.points[i];
            RotationMatrix R;
            R.SetFromEuler(orient.alpha, orient.beta, orient.gamma);
            RotationMatrix RT = R.Transpose(); // Lab to Body
            
            // Lab Polarization in Body Frame
            double eps_body[3] = {pol_lab[0], pol_lab[1], pol_lab[2]};
            RT.Apply(eps_body[0], eps_body[1], eps_body[2]);
            
            // Helper to compute Amplitude for a given k direction (Lab)
            auto ComputeAmp = [&](const double* k_lab) -> std::complex<double> {
                double k_body[3] = {k_lab[0], k_lab[1], k_lab[2]};
                RT.Apply(k_body[0], k_body[1], k_body[2]);
                
                // Theta, Phi of k_body
                double r = std::sqrt(k_body[0]*k_body[0] + k_body[1]*k_body[1] + k_body[2]*k_body[2]); 
                if (r < 1e-12) return 0.0;
                
                double theta_k = std::acos(k_body[2]/r);
                double phi_k = std::atan2(k_body[1], k_body[0]);
                double eta_k = std::cos(theta_k);
                
                std::complex<double> total_amp = 0.0;
                
                // Sum over mn
                for(const auto& elem : elements) {
                    // Check if solution exists
                    if (solutions.find(elem.m) == solutions.end()) continue;
                    const auto& sol = solutions[elem.m];
                    
                    // Evaluate Angular S_{mn}(k)
                    double S_val = phys.EvaluateAngular(eta_k, elem.m, elem.n_mode, sol);
                    
                    // Azimuthal Phase exp(-i m phi_k) / sqrt(2pi)
                    std::complex<double> phi_val = std::exp(std::complex<double>(0, -elem.m * phi_k));
                    phi_val /= std::sqrt(2.0 * M_PI);
                    
                    // Radial Phase (-i)^nu = exp(-i pi/2 nu)
                    std::complex<double> nu = elem.nu;
                    std::complex<double> phase = std::pow(std::complex<double>(0.0, -1.0), nu);
                    
                    // Matrix Element dot Polarization
                    std::complex<double> I_dot_eps = elem.I_x * eps_body[0] + elem.I_y * eps_body[1] + elem.I_z * eps_body[2];
                    
                    // Combine
                    total_amp += I_dot_eps * S_val * phi_val * phase;
                }
                return total_amp;
            };
            
            std::complex<double> A_par = ComputeAmp(k_par_lab);
            double sig_par = std::norm(A_par);
            
            std::complex<double> A_perp1 = ComputeAmp(k_perp1_lab);
            std::complex<double> A_perp2 = ComputeAmp(k_perp2_lab);
            double sig_perp = 0.5 * (std::norm(A_perp1) + std::norm(A_perp2));
            
            sum_sigma_par += sig_par * orient.weight;
            sum_sigma_perp += sig_perp * orient.weight;
        }
        
        // Finalize
        // Apply Norms (Squared, already applied via I_L * I_R approx in matrix element? 
        // No, we used I_avg = 0.5(I_L + I_R) in matrix elem.
        // So |I_avg|^2 approx |I_L||I_R|.
        // And we need dyson norms.
        double norms = dyson_L.qchem_norm * dyson_R.qchem_norm; 
        
        double sigma_par_total = sum_sigma_par * norms;
        double sigma_perp_total = sum_sigma_perp * norms;
        
        double denom = sigma_par_total + 2.0 * sigma_perp_total;
        double beta = 0.0;
        if (std::abs(denom) > 1e-15) {
            beta = 2.0 * (sigma_par_total - sigma_perp_total) / denom;
        }
        
        results.push_back({eKE, sigma_par_total, sigma_perp_total, beta});
    }
    
    return results;
}
