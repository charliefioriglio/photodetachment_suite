import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "production_data/evolution"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_verify(D, a, E, m, mode_idx, r_min, r_max, step, out_csv_name):
    full_path = os.path.join(OUTPUT_DIR, out_csv_name)
    # verify_radial Usage: D a E m mode_idx r_min r_max step output
    cmd = [
        "./verify_radial",
        str(D), str(a), str(E), str(m), str(mode_idx),
        str(r_min), str(r_max), str(step),
        full_path
    ]
    subprocess.run(cmd, check=True)
    return full_path

def main():
    print("Regenerating Evolution Plots...")
    
    # Common parameters inferred or standard
    E = 0.01 
    m = 0
    mode = 0
    
    # 1. Sweep D (Fixed a=2.0)
    print("--- Sweep D (a=2.0) ---")
    a = 2.0
    D_vals = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.0, 2.0, 5.0, 10.0] # Extended range
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(D_vals)))
    
    for i, D in enumerate(D_vals):
        # Format: S_a2.0_D0.05.csv
        # For D >= 1, format? D1.00?
        csv_name = f"S_a{a}_D{D:.2f}.csv" 
        
        # Max xi ~ 40? 
        r_max = 80.0 * a 
        r_min = a * 1.001
        
        path = run_verify(D, a, E, m, mode, r_min, r_max, 0.1*a, "temp.csv")
        
        # Read and Convert
        data = np.loadtxt(path, skiprows=1, delimiter=",")
        r = data[:,0]
        s_real = data[:,1]
        
        xi = r / a
        k = np.sqrt(2.0 * E)
        
        # Asymptotic Energy Normalization
        # We want S(r) ~ (1/kr) * sin(...) asymptotically.
        # So Envelope(S * kr) should be 1.
        # Calculate envelope of (S * kr) in the tail.
        # Use last 30% of points.
        
        scaled_s = s_real * (k * r)
        n_pts = len(scaled_s)
        tail_slice = scaled_s[int(n_pts * 0.7):]
        
        # Robust envelope estimate (max abs)
        env_amp = np.percentile(np.abs(tail_slice), 95) # Top 5% to avoid outliers/zeros
        if env_amp < 1e-6: env_amp = 1.0
        
        norm_factor = 1.0 / env_amp
        s_norm = s_real * norm_factor
        
        # Phase Condition: First Peak (|y|<0.5) -> Negative
        diffs = np.diff(s_norm)
        sign_diffs = np.sign(diffs)
        extrema = np.where(sign_diffs[:-1] != sign_diffs[1:])[0] + 1
        
        target_idx = -1
        for idx in extrema:
            if abs(s_norm[idx]) < 0.5:
                target_idx = idx
                break
        
        if target_idx != -1:
            if s_norm[target_idx] > 0: s_norm *= -1
            
        # Save exact CSV format expected
        df = pd.DataFrame({'xi': xi, 'S_real': s_norm, 'S_imag': np.zeros_like(s_norm)})
        df.to_csv(os.path.join(OUTPUT_DIR, csv_name), index=False, float_format="%.6g")
        
        plt.plot(xi, s_norm, color=colors[i], label=f"D={D}")
        
    plt.title(f"Evolution with Dipole Strength (a={a})")
    plt.xlabel("xi (r/a)")
    plt.ylabel("Radial Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 10) 
    plt.ylim(-1.5, 1.5)
    plt.savefig(os.path.join(OUTPUT_DIR, f"Sweep_D_a{a}.png"))
    plt.close()
    
    # 2. Sweep a (Fixed D=0.2)
    print("--- Sweep a (D=0.2) ---")
    D = 0.2
    a_vals = [1.00, 1.50, 2.00, 2.50, 3.00]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(a_vals)))
    
    for i, a_val in enumerate(a_vals):
        filename = f"S_D{D}_a{a_val:.2f}.csv" 
        
        r_max = 80.0 * a_val
        r_min = a_val * 1.001
        
        path = run_verify(D, a_val, E, m, mode, r_min, r_max, 0.1*a_val, "temp_a.csv")
        
        data = np.loadtxt(path, skiprows=1, delimiter=",")
        r = data[:,0]
        s_real = data[:,1]
        
        xi = r / a_val
        k = np.sqrt(2.0 * E)
        
        # Asymptotic Norm
        scaled_s = s_real * (k * r)
        n_pts = len(scaled_s)
        tail_slice = scaled_s[int(n_pts * 0.7):]
        env_amp = np.percentile(np.abs(tail_slice), 95)
        if env_amp < 1e-6: env_amp = 1.0
        
        norm_factor = 1.0 / env_amp
        s_norm = s_real * norm_factor
        
        # Phase Condition (First Peak < 0.5 -> Neg)
        diffs = np.diff(s_norm)
        sign_diffs = np.sign(diffs)
        extrema = np.where(sign_diffs[:-1] != sign_diffs[1:])[0] + 1
        
        target_idx = -1
        for idx in extrema:
            if abs(s_norm[idx]) < 0.5:
                target_idx = idx
                break
        
        if target_idx != -1:
            if s_norm[target_idx] > 0: s_norm *= -1
        
        df = pd.DataFrame({'xi': xi, 'S_real': s_norm, 'S_imag': np.zeros_like(s_norm)})
        df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False, float_format="%.6g")
        
        plt.plot(xi, s_norm, color=colors[i], label=f"a={a_val:.2f}")

    plt.title(f"Evolution with Length Scale (D={D})")
    plt.xlabel("xi (r/a)")
    plt.ylabel("Radial Function")
    plt.xlim(1, 10)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f"Sweep_a_D{D}.png"))
    plt.close()
    
    # Cleanup temps
    try:
        os.remove(os.path.join(OUTPUT_DIR, "temp.csv"))
        os.remove(os.path.join(OUTPUT_DIR, "temp_a.csv"))
    except: pass

if __name__ == "__main__":
    main()
