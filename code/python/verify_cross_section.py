import sys
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Add path to import dyson_io
sys.path.append(os.path.join(os.getcwd(), "src/python"))
import dyson_io

def verify_cuo():
    print("--- Verifying Physical Dipole Cross Section (CuO) ---")
    
    qchem_out = "reference materials/CuO_pVTZ.out"
    if not os.path.exists(qchem_out):
        print(f"Error: {qchem_out} not found.")
        return

    # 1. Parse Dyson Orbital
    print(f"Parsing {qchem_out}...")
    data = dyson_io.load_qchem(qchem_out)
    
    # Select Dyson 0 (usually the only one or primary)
    dyson_idx = [0] 
    
    # 2. Determine Grid (Standard padding)
    coords = np.array([a.center_bohr for a in data.atoms])
    # dyson_io centers internally in write_dyson usually? 
    # check load_qchem logic vs main. 
    # load_qchem returns raw coords from file.
    # main centers them.
    # We should center them too for grid calculation.
    centroid = np.mean(coords, axis=0)
    # We don't modify data.atoms in place yet, write_cpp_input does centering logic on copy?
    # No, dyson_io.write_cpp_input modifies atoms in place?
    # "coords = np.array([a.center_bohr...]) ... for a in atoms: a.center_bohr = ..."
    # Yes it modifies in place.
    # We can just let write_cpp_input handle it?
    # But write_cpp_input helper in dyson_io takes "data" and does centering.
    
    # We need to define grid bounds.
    # Centering happens inside write_cpp_input? 
    # Check dyson_io.py content...
    # Line 240: write_cpp_input(...)
    # Line 246: centroid = np.mean...
    # Line 249: for a in atoms: a.center_bohr = ... (Modifies object!)
    
    # Initial Coords for Grid bounds determination (assuming centering)
    # coords relative to centroid
    shifted_coords = coords - centroid
    padding = 20.0
    min_c = shifted_coords.min(axis=0) - padding
    max_c = shifted_coords.max(axis=0) + padding
    
    grid = {
        "x0": min_c[0], "x1": max_c[0],
        "y0": min_c[1], "y1": max_c[1],
        "z0": min_c[2], "z1": max_c[2],
        "step": 0.5
    }
    
    input_dat = "verification_dyson_input.dat"
    
    # Write Input
    dyson_io.write_cpp_input(data, dyson_idx, grid, input_dat)
    print(f"Generated {input_dat}")
    
    # 3. Setup Calculation Parameters
    D = 0.0
    a = 0.01
    
    # Energies: Single point test
    IE = 1.78
    eKEs = np.array([0.50816]) # Match one PWE point
    energies_ev = IE + eKEs
    energies_str = ",".join([f"{e:.4f}" for e in energies_ev])
    
    print(f"Running compute_xs (D={D}, a={a}, IE={IE})...")
    # ./compute_xs input_file D a energies IE
    cmd = ["./compute_xs", input_dat, str(D), str(a), energies_str, str(IE)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error Running compute_xs:")
        print(result.stderr)
        return
    
    # Parse Results
    print("Results:")
    lines = result.stdout.splitlines()
    xs_vals = []
    e_vals = []
    
    print(result.stdout) # Print raw for log
    
    for line in lines:
        if "," in line and "Energy" not in line:
            parts = line.split(",")
            try:
                e_val = float(parts[0])
                xs_val = float(parts[1])
                e_vals.append(e_val)
                xs_vals.append(xs_val)
            except ValueError:
                continue
            
    # Visualize
    e_vals = np.array(e_vals)
    xs_vals = np.array(xs_vals)
    eke_vals = e_vals - IE
    
    plt.figure(figsize=(8, 6))
    plt.plot(eke_vals, xs_vals, 'o-', label=f"Physical (D={D}, a={a})")
    plt.title("Physical Dipole Cross Section Verification (CuO)")
    plt.xlabel("eKE (eV)")
    plt.ylabel("Cross Section (au)")
    plt.grid(True)
    plt.legend()
    plt.savefig("verification_cuo_plot.png")
    print("Saved verification_cuo_plot.png")

if __name__ == "__main__":
    verify_cuo()
