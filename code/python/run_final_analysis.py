import json
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def run_job(config, filename):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Running job: {filename}")
    subprocess.check_call(["python3", "src/python/run_job.py", filename])

def main():
    root_dir = os.getcwd()
    output_dir = os.path.join(root_dir, "production_data/final_comparison")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    qchem_out = os.path.join(root_dir, "reference materials/CuO_pVTZ.out")
    
    # Energy Grid (0.1 to 2.1 eV eKE -> Photon Energy = eKE + 1.778)
    eKEs = np.linspace(0.1, 2.1, 20).tolist()
    
    base_config = {
        "qchem_output": qchem_out,
        "dyson": {
            "do_generation": True,
            "indices": [0, 1],
            "grid_step": 0.3,
            "padding": 20.0,
            "output_bin": os.path.join(output_dir, "dyson.bin")
        },
        "calculation": {
            "do_calculation": True,
            "skip_beta_gen": True,
            "type": "cross_section",
            "ie": 1.778,
            "energies": eKEs,
            "l_max": 4,
            "points": 50
        },
        "visualization": {"do_plot": False}
    }
    
    jobs = []
    
    # 1. PWE
    cfg = base_config.copy()
    cfg["calculation"] = base_config["calculation"].copy()
    cfg["calculation"]["model"] = "pwe"
    cfg["calculation"]["output_csv"] = os.path.join(output_dir, "xs_pwe.csv")
    jobs.append(("PWE", cfg))
    
    # 2. Point Dipole (D=0.4)
    cfg = base_config.copy()
    cfg["calculation"] = base_config["calculation"].copy()
    cfg["calculation"]["model"] = "point_dipole"
    cfg["calculation"]["dipole"] = 0.4
    cfg["calculation"]["output_csv"] = os.path.join(output_dir, "xs_pd_D0.4.csv")
    jobs.append(("Point Dipole (D=0.4)", cfg))
    
    # 3. Physical Dipole (D=[0, 0.4, 1], a=[0.1, 2])
    dipoles = [0.0, 0.4, 1.0]
    lengths = [0.1, 2.0]
    
    for D in dipoles:
        for a in lengths:
            cfg = base_config.copy()
            cfg["calculation"] = base_config["calculation"].copy()
            cfg["calculation"]["model"] = "physical_dipole"
            cfg["calculation"]["dipole"] = D
            cfg["calculation"]["dipole_length"] = a
            
            # Label
            label = f"Physical Dipole (D={D}, a={a})"
            fname = f"xs_phys_D{D}_a{a}.csv"
            cfg["calculation"]["output_csv"] = os.path.join(output_dir, fname)
            
            jobs.append((label, cfg))
            
    # Run Jobs
    results = {}
    json_path = os.path.join(output_dir, "temp_job.json")
    
    # Optimize: Only generate Dyson once?
    # run_job.py re-runs dyson generation every time if "do_generation": true.
    # We can set "do_generation": false for subsequent jobs if we point to same bin.
    # But dyson_io uses args to write input file. So we need to run it.
    # Actually, dyson_io handles the logic. 
    # It constructs input.dat.
    
    for i, (label, config) in enumerate(jobs):
        # reuse binary if possible to save time?
        # But dyson_io rewrites binary every time.
        # Let's just run it. 20 pts is fast.
        
        # Unique bin for safety or just overwrite? Overwrite is fine.
        config["dyson"]["output_bin"] = os.path.join(output_dir, "dyson.bin")
        
        print(f"--- Processing {label} ---")
        run_job(config, json_path)
        
        # Load result
        csv_path = config["calculation"]["output_csv"]
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Add eKE
            df["eKE"] = df["Energy"] - 1.778
            results[label] = df
        else:
            print(f"Warning: Output {csv_path} not found.")

    # Plotting
    print("Generating Plot...")
    plt.figure(figsize=(10, 8))
    
    markers = ['o', 's', '^', 'v', 'D', 'x', '+', '*']
    colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    i = 0
    for label, df in results.items():
        # Sort by eKE just in case
        df = df.sort_values("eKE")
        plt.plot(df["eKE"], df["CrossSection"], 
                 marker=markers[i % len(markers)], 
                 color=colors[i % len(colors)],
                 label=label, 
                 linestyle='-', 
                 markersize=4, alpha=0.8)
        i += 1
        
    plt.xlabel("Electron Kinetic Energy (eV)")
    plt.ylabel("Cross Section (a.u.)")
    plt.title("Cross Section Comparison: PWE vs Point Dipole vs Physical Dipole")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join(output_dir, "final_comparison_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
