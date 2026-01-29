
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import sys
from scipy.special import factorial
from scipy.linalg import eigh

# --- Reference Implementation (from angular.py) ---
def analytic(m, L_max, E, a, D):
    m = np.abs(m)
    c = np.sqrt(2 * E * a**2)
    ell_vals = np.arange(m, L_max + 1)
    N = len(ell_vals)
    H = np.zeros((N, N))
    S_diag = np.array([2 * factorial(l + m) / ((2 * l + 1) * factorial(l - m)) for l in ell_vals])
    S = np.diag(S_diag)

    for i, l in enumerate(ell_vals):
        if l - m >= 0:
            f1 = -(l * (l + 1)) * S_diag[i]
            f2 = -c**2 * (
                (l + m) * (l - m) / ((2 * l + 1) * (2 * l - 1)) +
                (l - m + 1) * (l + m + 1) / ((2 * l + 1) * (2 * l + 3))
            ) * S_diag[i]
            H[i, i] += f1 + f2

        if i + 1 < N:
            f = (-2 * D / (2 * l + 1)) * (l + 1 - m)
            f *= 2 * factorial(l + m + 1) / ((2 * l + 3) * factorial(l - m + 1))
            H[i + 1, i] += f
            H[i, i + 1] += f
        if i - 1 >= 0:
            f = (-2 * D / (2 * l + 1)) * (l + m)
            f *= 2 * factorial(l + m - 1) / ((2 * l - 1) * factorial(l - m - 1))
            H[i - 1, i] += f
            H[i, i - 1] += f

        if i + 2 < N:
            f = -c**2 * (l - m + 1) * (l - m + 2)
            f *= 2 * factorial(l + m + 2) / (
                (2 * l + 1) * (2 * l + 3) * (2 * l + 5) * factorial(l - m + 2)
            )
            H[i + 2, i] += f
            H[i, i + 2] += f
        if i - 2 >= 0:
            f = -c**2 * (l + m) * (l + m - 1)
            f *= 2 * factorial(l + m - 2) / (
                (2 * l + 1) * (2 * l - 1) * (2 * l - 3) * factorial(l - m - 2)
            )
            H[i - 2, i] += f
            H[i, i - 2] += f

    eigvals, eigvecs = eigh(H, S)
    # Reverse to match "Descending" expectations (?) check later
    # Reference code did [::-1]
    eigvals = eigvals[::-1]
    return eigvals

# --- Run C++ Driver ---
def run_cpp_test(m, l_max, E, a, D, plot_file="temp_plot.csv"):
    cmd = ["./test_angular", str(m), str(l_max), str(E), str(a), str(D), plot_file]
    try:
        res = subprocess.check_output(cmd, text=True)
        return res
    except subprocess.CalledProcessError as e:
        print("Error running C++:", e)
        return ""

# --- Main ---
def main():
    if not os.path.exists("production_data/physical_dipole"):
        os.makedirs("production_data/physical_dipole")

    cases = [
        {"desc": "Base Case", "m": 0, "l_max": 10, "E": 0.003676, "a": 0.0, "D": 0.34},
        {"desc": "Vary D",    "m": 0, "l_max": 10, "E": 0.003676, "a": 0.0, "D": 1.5},
        {"desc": "Vary a",    "m": 0, "l_max": 10, "E": 0.003676, "a": 2.0, "D": 0.5},
        {"desc": "m=1",       "m": 1, "l_max": 10, "E": 0.003676, "a": 0.0, "D": 0.34}
    ]
    
    for case in cases:
        print(f"\nRunning Case: {case['desc']}")
        m, l_max, E, a, D = case["m"], case["l_max"], case["E"], case["a"], case["D"]
        plot_csv = f"production_data/physical_dipole/angular_{case['desc'].replace(' ', '_')}.csv"
        
        # 1. Run C++
        output = run_cpp_test(m, l_max, E, a, D, plot_csv)
        
        # Parse Eigenvalues from C++ output
        cpp_evals = []
        for line in output.splitlines():
            if line.startswith("Eig "):
                cpp_evals.append(float(line.split(":")[1].strip()))
        
        # 2. Run Analytic Python
        py_evals = analytic(m, l_max, E, a, D)
        
        # 3. Compare
        print(f"{'Index':<5} {'C++':<15} {'Python':<15} {'Diff':<15}")
        for i in range(min(len(cpp_evals), len(py_evals), 5)):
             diff = abs(cpp_evals[i] - py_evals[i])
             print(f"{i:<5} {cpp_evals[i]:<15.8f} {py_evals[i]:<15.8f} {diff:<15.2e}")
             
        # 4. Plot Angular Functions
        if os.path.exists(plot_csv):
            import pandas as pd
            df = pd.read_csv(plot_csv)
            plt.figure(figsize=(8,5))
            # Plot State 0 (Ground/Highest???) 
            # Note: In angular.py, eigvals are reversed (High to Low?).
            # Usually lambda = l(l+1) > 0.
            # Lambda increases with l.
            # If sorted High to Low, Index 0 is Highest Lambda.
            # Index N-1 is Lowest Lambda (Ground State).
            # Let's plot Index 0 and Index N-1 (if N>1)
            
            cols = [c for c in df.columns if c.startswith("State_")]
            
            for col in cols:
                plt.plot(df["Eta"], df[col], label=f"{col}")
                
            plt.xlabel(r"$\eta$")
            plt.ylabel(r"$T(\eta)$")
            plt.title(f"Angular Functions: {case['desc']} (m={m}, D={D}, a={a})")
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_csv.replace(".csv", ".png"))
            plt.close()
            print(f"Saved plot: {plot_csv.replace('.csv', '.png')}")

if __name__ == "__main__":
    main()
