import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

OUTPUT_DIR = "production_data/physical_dipole_final_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_verify(D, a, E, m, mode_idx, r_min, r_max, step, name):
    outfile = os.path.join(OUTPUT_DIR, f"{name}.csv")
    if os.path.exists(outfile): return outfile
    
    cmd = [
        "./verify_radial",
        str(D), str(a), str(E), str(m), str(mode_idx),
        str(r_min), str(r_max), str(step),
        outfile
    ]
    # print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return outfile

def load_data(csv_file):
    return np.loadtxt(csv_file, skiprows=1, delimiter=",")

def normalize(r, s, E):
    # Asymptotic Energy Normalization
    k = np.sqrt(2.0 * E)
    scaled = s * (k * r)
    
    n = len(scaled)
    tail = scaled[int(n * 0.6):] 
    amp = np.percentile(np.abs(tail), 95)
    if amp < 1e-6: amp = 1.0 
    
    s_norm = s / amp
    
    # Phase Condition: First peak with magnitude < 0.5 must be NEGATIVE.
    # Logic:
    # 1. Find all extrema.
    # 2. Iterate. If abs(extremum) < 0.5, this is our "sign determinant".
    # 3. If this extremum is Positive, flip the whole function.
    
    diffs = np.diff(s_norm)
    sign_diffs = np.sign(diffs)
    # Indices where sign of derivative changes
    extrema_indices = np.where(sign_diffs[:-1] != sign_diffs[1:])[0] + 1
    
    target_idx = -1
    for idx in extrema_indices:
        val = s_norm[idx]
        # Check magnitude
        if abs(val) < 0.5:
            target_idx = idx
            break
            
    if target_idx != -1:
        # We found a "small" peak/trough.
        # Ensure it is Negative.
        if s_norm[target_idx] > 0:
            s_norm *= -1
    else:
        # Fallback if no peaks < 0.5 found (unlikely for far-field normalized functions,
        # unless they are purely large oscillations > 0.5 which is fine, or just noise).
        # We might default to standard "First Major Peak Positive" or something?
        # User implies there WILL be such a peak. 
        # If not, maybe we just leave it or use the start value.
        # Let's fallback to checking start value negative for consistency?
        # Or just leave as is.
        pass
            
    return s_norm
            
    return s_norm

def plot_set(data_dict, title, filename_base, xlabel="r (Bohr)"):
    # 1. Full Range Plot (Zoom Out)
    plt.figure(figsize=(10, 6))
    for label, (x, y, color, style) in data_dict.items():
        plt.plot(x, y, linestyle=style, color=color, label=label, linewidth=1.5)
    
    plt.title(f"{title} (Far Field)")
    plt.xlabel(xlabel)
    plt.ylabel("Radial Function (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 80)
    plt.ylim(-2.0, 2.0) # Slightly looser to see peaks
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}_far.png"))
    plt.close()
    
    # 2. Near Field Plot (Zoom In)
    plt.figure(figsize=(10, 6))
    for label, (x, y, color, style) in data_dict.items():
        plt.plot(x, y, linestyle=style, color=color, label=label, linewidth=2.0)
        
    plt.title(f"{title} (Near Field)")
    plt.xlabel(xlabel)
    plt.ylabel("Radial Function (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 15) 
    plt.ylim(-2.0, 2.0)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}_near.png"))
    plt.close()

def main():
    print("Starting Comprehensive Physical Dipole Verification...")
    E_au = 0.01 # ~0.27 eV
    k = np.sqrt(2*E_au)
    step = 0.05
    
    # ==========================================
    # 1. Limiting Cases (D=0, small a)
    # ==========================================
    print("Generating: Limiting Cases...")
    
    D = 1e-6
    m = 0
    mode = 0
    
    f_a1 = run_verify(D, 1.0, E_au, m, mode, 1.0, 100.0, step, "limit_D0_a1")
    f_a0 = run_verify(D, 0.001, E_au, m, mode, 0.001, 100.0, step, "limit_D0_a0")
    
    d_a1 = load_data(f_a1)
    d_a0 = load_data(f_a0)
    
    r1, s1 = d_a1[:,0], normalize(d_a1[:,0], d_a1[:,1], E_au)
    r0, s0 = d_a0[:,0], normalize(d_a0[:,0], d_a0[:,1], E_au)
    rp, sp = d_a1[:,0], d_a1[:,4] # PWE Column
    
    data = {
        "Physical a=1.0": (r1, s1, "red", "-"),
        "Physical a=0.001": (r0, s0, "green", "--"),
        "Exact PWE (j0)": (rp, sp, "black", ":")
    }
    plot_set(data, "Zero Field Limit (D=0, m=0)", "limits_D0_m0")
    
    # ==========================================
    # 2. Finite Dipole Limits (Phys D=0.25 vs Point D=0.5)
    # ==========================================
    print("Generating: Finite Dipole Limits...")
    # User Request: Phys D=0.25, Point D=0.5.
    
    # 1. Generate Physical Dipole Data (D=0.25)
    D_phys = 0.25
    f_phys_a1 = run_verify(D_phys, 1.0, E_au, m, mode, 1.0, 100.0, step, "limit_physD0.25_a1")
    f_phys_a0 = run_verify(D_phys, 0.001, E_au, m, mode, 0.001, 100.0, step, "limit_physD0.25_a0")
    
    # 2. Generate Point Dipole Data (D=0.5)
    D_point = 0.5
    f_point = run_verify(D_point, 1.0, E_au, m, mode, 1.0, 100.0, step, "limit_pointD0.5_ref")
    
    d_a1 = load_data(f_phys_a1)
    d_a0 = load_data(f_phys_a0)
    d_ref = load_data(f_point)
    
    # Normalize everything using the same Phase Convention
    r1, s1 = d_a1[:,0], normalize(d_a1[:,0], d_a1[:,1], E_au)
    r0, s0 = d_a0[:,0], normalize(d_a0[:,0], d_a0[:,1], E_au)
    
    # Use column 3 (Point Dipole) from the Point Dipole run
    rp, sp = d_ref[:,0], normalize(d_ref[:,0], d_ref[:,3], E_au) 
    
    data = {
        "Physical a=1.0 (D=0.25)": (r1, s1, "red", "-"),
        "Physical a=0.001 (D=0.25)": (r0, s0, "green", "--"),
        "Point Dipole (D=0.5)": (rp, sp, "black", ":")
    }
    plot_set(data, f"Finite Dipole Comparison (Phys D={D_phys} vs Point D={D_point})", "limits_finite_dipole")

    # ==========================================
    # 3. Dipole Strength Sweep (Focused Range 0-1)
    # ==========================================
    print("Generating: Dipole Sweep...")
    a = 2.0
    # Even distributions 0 to 1
    D_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    cmap = cm.turbo
    norm_c = mcolors.Normalize(vmin=0, vmax=len(D_vals)-1)
    
    data = {}
    for i, val in enumerate(D_vals):
        fname = run_verify(val, a, E_au, 0, 0, a, 150.0, step, f"sweep_D_{val}")
        d = load_data(fname)
        r, s = d[:,0], d[:,1]
        
        # Plot vs xi = r/a
        xi = r / a
        s_norm = normalize(r, s, E_au)
        
        color = cmap(norm_c(i))
        data[f"D={val}"] = (xi, s_norm, color, "-")
        
    plot_set(data, f"Dipole Strength Evolution (a={a}, m=0)", "sweep_D_xi", xlabel="xi (r/a)")

    # ==========================================
    # 4. Length Scale Sweep (Super-Critical)
    # ==========================================
    print("Generating: Length Scale Sweep...")
    D = 0.5
    a_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    data = {}
    norm_c = mcolors.Normalize(vmin=0, vmax=len(a_vals)-1)
    
    for i, val in enumerate(a_vals):
        fname = run_verify(D, val, E_au, 0, 0, 0.1, 200.0, step, f"sweep_a_{val}")
        d = load_data(fname)
        r, s = d[:,0], d[:,1]
        
        # User request: Revert to c*xi (kr) scaling for a-sweep
        cxi = k * r
        
        s_norm = normalize(r, s, E_au)
        
        color = cmap(norm_c(i))
        data[f"a={val}"] = (cxi, s_norm, color, "-")
        
    plot_set(data, f"Length Scale Evolution (D={D}, m=0)", "sweep_a_cxi", xlabel="kr (c*xi)")

    # ==========================================
    # 5. Higher Angular Momentum (m Sweep)
    # ==========================================
    print("Generating: m Sweep...")
    D = 1.0
    a = 2.0
    m_vals = [0, 1, 2, 3, 4]
    
    data = {}
    norm_c = mcolors.Normalize(vmin=0, vmax=len(m_vals)-1)
    
    for i, m_val in enumerate(m_vals):
        fname = run_verify(D, a, E_au, m_val, 0, a, 100.0, step, f"sweep_m_{m_val}")
        d = load_data(fname)
        r, s = d[:,0], d[:,1]
        
        xi = r / a
        s_norm = normalize(r, s, E_au)
        
        color = cmap(norm_c(i))
        data[f"m={m_val}"] = (xi, s_norm, color, "-")
        
    plot_set(data, f"Azimuthal Mode Evolution (D={D}, a={a})", "sweep_m_xi", xlabel="xi (r/a)")

    # ==========================================
    # 6. Higher Radial Modes (mode/l Sweep)
    # ==========================================
    print("Generating: Mode (n/l) Sweep...")
    # For a given m, solve for n=0, 1, 2... (roughly l=m, m+1, m+2...)
    D = 1.0
    a = 2.0
    m = 0
    modes = [0, 1, 2, 3] # n=0,1,2,3
    
    data = {}
    norm_c = mcolors.Normalize(vmin=0, vmax=len(modes)-1)
    
    for i, n_val in enumerate(modes):
        fname = run_verify(D, a, E_au, m, n_val, a, 100.0, step, f"sweep_mode_n{n_val}")
        d = load_data(fname)
        r, s = d[:,0], d[:,1]
        
        xi = r / a
        s_norm = normalize(r, s, E_au)
        
        # PWE Check? n=0 -> l=0 (s-wave), n=1 -> l=1 (p-wave)...
        
        color = cmap(norm_c(i))
        data[f"n={n_val} (~l={n_val})"] = (xi, s_norm, color, "-")
        
    plot_set(data, f"Radial Mode Evolution (D={D}, a={a}, m={m})", "sweep_n_xi", xlabel="xi (r/a)")

    print("Done. Plots saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
