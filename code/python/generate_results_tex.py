import pandas as pd
import os

root_dir = "production_data/final_comparison"
files = [
    ("PWE", "xs_pwe.csv"),
    ("Point Dipole ($D=0.4$)", "xs_pd_D0.4.csv"),
    ("Phys ($D=0, a=0.1$)", "xs_phys_D0.0_a0.1.csv"),
    ("Phys ($D=0, a=2.0$)", "xs_phys_D0.0_a2.0.csv"),
    ("Phys ($D=0.4, a=0.1$)", "xs_phys_D0.4_a0.1.csv"),
    ("Phys ($D=0.4, a=2.0$)", "xs_phys_D0.4_a2.0.csv"),
    ("Phys ($D=1.0, a=0.1$)", "xs_phys_D1.0_a0.1.csv"),
    ("Phys ($D=1.0, a=2.0$)", "xs_phys_D1.0_a2.0.csv"),
]

# Read data
data = {}
energies = []
for label, fname in files:
    path = os.path.join(root_dir, fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        # eKE = Energy - 1.778. Assume rows are sorted.
        # Let's extract specific indices: 0 (low), 9 (mid), 19 (high) if 20 pts
        indices = [0, 9, 19]
        vals = []
        for idx in indices:
            if idx < len(df):
                row = df.iloc[idx]
                eKE = row['Energy'] - 1.778
                xs = row['CrossSection']
                vals.append((eKE, xs))
        data[label] = vals
        if not energies:
            energies = [v[0] for v in vals]

# Generate LaTeX Table
print(r"\section{Results and Comparison}")
print(r"")
print(r"\subsection{Cross Section Comparison Table}")
print(r"")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{|l|c|c|c|}")
print(r"\hline")
print(r"Model & \multicolumn{3}{c|}{Cross Section (a.u.) at selected eKE} \\")
print(r"\cline{2-4}")
headers = [f"{e:.2f} eV" for e in energies]
print(f" & {headers[0]} & {headers[1]} & {headers[2]} \\\\")
print(r"\hline")

for label, fname in files:
    if label in data:
        vals = data[label]
        row_str = f"{label} & {vals[0][1]:.4f} & {vals[1][1]:.4f} & {vals[2][1]:.4f} \\\\"
        print(row_str)
print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Comparison of total cross sections for different continuum models at low, medium, and high electron kinetic energies.}")
print(r"\label{tab:xs_comparison}")
print(r"\end{table}")

print(r"")
print(r"\subsection{Physical Dipole Radial Functions}")
print(r"")

# Plots
# 1. Sweep D (Near/Far)
print(r"\begin{figure}[H]")
print(r"    \centering")
print(r"    \includegraphics[width=0.48\textwidth]{production_data/physical_dipole_final_plots/sweep_D_xi_near.png}")
print(r"    \includegraphics[width=0.48\textwidth]{production_data/physical_dipole_final_plots/sweep_D_xi_far.png}")
print(r"    \caption{Radial functions $S_{00}(\xi)$ for varying dipole strength $D$. Left: Near-field ($\xi \approx 1$). Right: Far-field ($\xi \gg 1$).}")
print(r"    \label{fig:sweep_D}")
print(r"\end{figure}")

# 2. Sweep m
print(r"\begin{figure}[H]")
print(r"    \centering")
print(r"    \includegraphics[width=0.6\textwidth]{production_data/physical_dipole_final_plots/sweep_m_xi_near.png}")
print(r"    \caption{Radial functions for different magnetic quantum numbers $m$, showing the centrifugal barrier effect near the origin.}")
print(r"    \label{fig:sweep_m}")
print(r"\end{figure}")

# 3. Limits
print(r"\begin{figure}[H]")
print(r"    \centering")
print(r"    \includegraphics[width=0.48\textwidth]{production_data/physical_dipole_final_plots/limits_finite_dipole_far.png}")
print(r"    \includegraphics[width=0.48\textwidth]{production_data/physical_dipole_final_plots/limits_D0_m0_far.png}")
print(r"    \caption{Limiting behavior verification. Left: Comparison of finite dipole result with asymptotic form. Right: $D=0, m=0$ limit comparing with Spherical Bessel $j_0$.}")
print(r"    \label{fig:limits}")
print(r"\end{figure}")

# 4. Final Comparison
print(r"\begin{figure}[H]")
print(r"    \centering")
print(r"    \includegraphics[width=0.8\textwidth]{production_data/final_comparison/final_comparison_plot.png}")
print(r"    \caption{Total photodetachment cross section of CuO. Comparison between Plane Wave Expansion (PWE), Point Dipole ($D=0.4$), and Physical Dipole models with varying parameters.}")
print(r"    \label{fig:final_comparison}")
print(r"\end{figure}")
