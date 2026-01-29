#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace LinearAlgebra {

// Symmetric formulation
// Computes eigenvalues and eigenvectors of a real symmetric matrix A.
// A is n x n.
// w returns eigenvalues.
// V returns eigenvectors (columns).
inline void Jacobi(int n, std::vector<std::vector<double>> A, std::vector<double>& w, std::vector<std::vector<double>>& V) {
    const int max_iter = 100;
    const double eps = 1e-12;
    
    // Initialize V to identity
    V.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) V[i][i] = 1.0;
    
    // Initialize w to diagonal of A
    w.resize(n);
    std::vector<double> b(n), z(n);
    for (int i = 0; i < n; i++) {
        w[i] = A[i][i];
        b[i] = A[i][i];
        z[i] = 0.0;
    }
    
    for (int iter = 0; iter < max_iter; iter++) {
        double sm = 0.0;
        for (int ip = 0; ip < n - 1; ip++) {
            for (int iq = ip + 1; iq < n; iq++) {
                sm += std::abs(A[ip][iq]);
            }
        }
        
        if (sm < eps) return; // Converged
        
        double tresh;
        if (iter < 4) tresh = 0.2 * sm / (n * n);
        else tresh = 0.0;
        
        for (int ip = 0; ip < n - 1; ip++) {
            for (int iq = ip + 1; iq < n; iq++) {
                double g = 100.0 * std::abs(A[ip][iq]);
                if (iter > 4 && (std::abs(w[ip]) + g == std::abs(w[ip]))
                    && (std::abs(w[iq]) + g == std::abs(w[iq]))) {
                    A[ip][iq] = 0.0;
                } else if (std::abs(A[ip][iq]) > tresh) {
                    double h = w[iq] - w[ip];
                    double t;
                    if (std::abs(h) + g == std::abs(h)) {
                        t = (A[ip][iq]) / h;
                    } else {
                        double theta = 0.5 * h / (A[ip][iq]);
                        t = 1.0 / (std::abs(theta) + std::sqrt(1.0 + theta * theta));
                        if (theta < 0.0) t = -t;
                    }
                    double c = 1.0 / std::sqrt(1 + t * t);
                    double s = t * c;
                    double tau = s / (1.0 + c);
                    h = t * A[ip][iq];
                    z[ip] -= h;
                    z[iq] += h;
                    w[ip] -= h;
                    w[iq] += h;
                    A[ip][iq] = 0.0;
                    
                    for (int j = 0; j <= ip - 1; j++) {
                        double g_val = A[j][ip];
                        double h_val = A[j][iq];
                        A[j][ip] = g_val - s * (h_val + g_val * tau);
                        A[j][iq] = h_val + s * (g_val - h_val * tau);
                    }
                    for (int j = ip + 1; j <= iq - 1; j++) {
                        double g_val = A[ip][j];
                        double h_val = A[j][iq];
                        A[ip][j] = g_val - s * (h_val + g_val * tau);
                        A[j][iq] = h_val + s * (g_val - h_val * tau);
                    }
                    for (int j = iq + 1; j < n; j++) {
                        double g_val = A[ip][j];
                        double h_val = A[iq][j];
                        A[ip][j] = g_val - s * (h_val + g_val * tau);
                        A[iq][j] = h_val + s * (g_val - h_val * tau);
                    }
                    for (int j = 0; j < n; j++) {
                        double g_val = V[j][ip];
                        double h_val = V[j][iq];
                        V[j][ip] = g_val - s * (h_val + g_val * tau);
                        V[j][iq] = h_val + s * (g_val - h_val * tau);
                    }
                }
            }
        }
        for (int i = 0; i < n; i++) {
            b[i] += z[i];
            w[i] = b[i];
            z[i] = 0.0;
        }
    }
}

} // namespace LinearAlgebra

#endif // LINALG_H
