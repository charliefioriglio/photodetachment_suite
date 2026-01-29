import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

OUTPUT_DIR = "production_data/physical_dipole_final_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_radial(D, a, E, m, mode_idx, r_min, r_max, step, name):
    outfile = os.path.join(OUTPUT_DIR, f"{name}.csv")
    cmd = [
        "./verify_radial",
        str(D), str(a), str(E), str(m), str(mode_idx),
        str(r_min), str(r_max), str(step),
        outfile
    ]
    subprocess.run(cmd, check=True)
    return np.loadtxt(outfile, skiprows=1, delimiter=",")

def run_angular(D, a, E, m, l_target, num_pts, name):
    outfile = os.path.join(OUTPUT_DIR, f"{name}.csv")
    cmd = [
        "./verify_angular",
        str(D), str(a), str(E), str(m), str(l_target),
        str(num_pts),
        outfile
    ]
    subprocess.run(cmd, check=True)
    return np.loadtxt(outfile, skiprows=1, delimiter=",")

def plot_radial_compare():
    print("Generating Radial Comparisons (D=0, a=0.01)...")
    # For D=0, Physical should match Bessel j_l.
    # a must be small for coordinates to match (xi = r/a -> r)?
    # ACTUALLY: For D=0, the equation is independent of 'a' if solved in r?
    # No, it's solved in xi. xi = r/a.
    # The equation for S(xi) depends on c = k*a.
    # If a is small, c is small.
    # The solution S(xi) mapped to r should be j_l(k*r).
    
    D = 0.0
    a = 0.01
    E = 0.5 # Higher energy to see oscillations
    
    # --- Case 1: l=1, m=0 ---
    # mode_idx? l starts at |m|=0. l=0 (idx0), l=1 (idx1).
    # so mode_idx = 1.
    m = 0
    mode = 1 # l=1
    target_l = 1
    
    d1 = run_radial(D, a, E, m, mode, 0.1, 40.0, 0.1, "compare_rad_l1_m0")
    
    r = d1[:,0]
    s_phys = d1[:,1] # Real part
    s_pwe = d1[:,4]  # PWE Column (j_l)
    
    # Normalize by amplitude to compare shapes
    s_phys_norm = s_phys / np.max(np.abs(s_phys))
    s_pwe_norm = s_pwe / np.max(np.abs(s_pwe))
    
    # Check phase
    if np.sign(s_phys_norm[5]) != np.sign(s_pwe_norm[5]):
        s_phys_norm *= -1
        
    plt.figure()
    plt.plot(r, s_phys_norm, 'r-', label='Physical (D=0)', lw=2)
    plt.plot(r, s_pwe_norm, 'k--', label='PWE j_1(kr)')
    plt.title(f"Radial Comparison l=1, m=0 (D={D})")
    plt.legend()
    plt.xlabel("r")
    plt.savefig(f"{OUTPUT_DIR}/compare_rad_l1_m0.png")
    plt.close()

    # --- Case 2: l=1, m=1 ---
    # l starts at |m|=1. l=1 is idx0.
    m = 1
    mode = 0 # l=1
    target_l = 1
    
    d2 = run_radial(D, a, E, m, mode, 0.1, 40.0, 0.1, "compare_rad_l1_m1")
    
    r = d2[:,0]
    s_phys = d2[:,1]
    s_pwe = d2[:,4]
    
    s_phys_norm = s_phys / np.max(np.abs(s_phys))
    s_pwe_norm = s_pwe / np.max(np.abs(s_pwe))
    
    if np.sign(s_phys_norm[5]) != np.sign(s_pwe_norm[5]):
        s_phys_norm *= -1
        
    plt.figure()
    plt.plot(r, s_phys_norm, 'r-', label='Physical (D=0)', lw=2)
    plt.plot(r, s_pwe_norm, 'k--', label='PWE j_1(kr)')
    plt.title(f"Radial Comparison l=1, m=1 (D={D})")
    plt.legend()
    plt.xlabel("r")
    plt.savefig(f"{OUTPUT_DIR}/compare_rad_l1_m1.png")
    plt.close()

def plot_angular_compare():
    print("Generating Angular Comparisons (D=0)...")
    D = 0.0
    a = 1.0 # a shouldn't matter for Angular D=0 (Spherical limit)? 
            # Except c = k*a enters separation constant. 
            # For H atom / PWE, c=0 effectively? 
            # Or if we use Prolate functions for finite k*a, they deviate from Legendre?
            # User wants to check consistency. With D=0, Physical Dipole eq becomes Spheroidal Wave Eq.
            # If c -> 0, it becomes Legendre.
            # So use small a or small E.
    E = 0.01 
    a = 0.01 
    num_pts = 100
    
    # --- Case 1: l=0, m=0 ---
    m = 0
    l = 0
    d1 = run_angular(D, a, E, m, l, num_pts, "compare_ang_l0_m0")
    
    eta = d1[:,0]
    t_phys = d1[:,2]
    y_lm = d1[:,3]
    
    # Normalize max to 1
    t_phys /= np.max(np.abs(t_phys))
    y_lm /= np.max(np.abs(y_lm))
    
    plt.figure()
    plt.plot(eta, t_phys, 'r-', label='Physical T(eta)')
    plt.plot(eta, y_lm, 'k--', label='Y_00 theta part')
    plt.title(f"Angular Comparison l=0, m=0")
    plt.legend()
    plt.xlabel("cos(theta)")
    plt.savefig(f"{OUTPUT_DIR}/compare_ang_l0_m0.png")
    plt.close()
    
    # --- Case 2: l=1, m=0 ---
    m = 0
    l = 1
    d2 = run_angular(D, a, E, m, l, num_pts, "compare_ang_l1_m0")
    
    eta = d2[:,0]
    t_phys = d2[:,2]
    y_lm = d2[:,3]
    
    t_phys /= np.max(np.abs(t_phys))
    y_lm /= np.max(np.abs(y_lm))
    
    # Check Phase (Y_10 ~ cos theta = eta. Odd.)
    # If one is flipped
    
    plt.figure()
    plt.plot(eta, t_phys, 'r-', label='Physical T(eta)')
    plt.plot(eta, y_lm, 'k--', label='Y_10 theta part')
    plt.title(f"Angular Comparison l=1, m=0")
    plt.legend()
    plt.xlabel("cos(theta)")
    plt.savefig(f"{OUTPUT_DIR}/compare_ang_l1_m0.png")
    plt.close()
    
    # --- Case 3: l=1, m=1 ---
    m = 1
    l = 1
    d3 = run_angular(D, a, E, m, l, num_pts, "compare_ang_l1_m1")
    
    eta = d3[:,0]
    t_phys = d3[:,2]
    y_lm = d3[:,3]
    
    t_phys /= np.max(np.abs(t_phys))
    y_lm /= np.max(np.abs(y_lm))
    
    plt.figure()
    plt.plot(eta, t_phys, 'r-', label='Physical T(eta)')
    plt.plot(eta, y_lm, 'k--', label='Y_11 theta part')
    plt.title(f"Angular Comparison l=1, m=1")
    plt.legend()
    plt.xlabel("cos(theta)")
    plt.savefig(f"{OUTPUT_DIR}/compare_ang_l1_m1.png")
    plt.close()

if __name__ == "__main__":
    plot_radial_compare()
    plot_angular_compare()
    print("Done.")
