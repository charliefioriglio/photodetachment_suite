
import os
import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- Configuration ---
OUTPUT_DIR = "production_data/CuO cross sections"
RUN_JOB_SCRIPT = "src/python/run_job.py"
PVTZ_OUT = "reference materials/CuO_pVTZ.out"

# Params
IE_BASE = 1.778
L_MAX = 4
GRID_STEP = 0.3
PADDING = 20.0
# Dipoles
DIPOLES_ALL = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
DIPOLES_ZOOM = [0.0, 0.1, 0.2, 0.4, 0.6]

# --- Helpers ---
def run_job(config, job_name):
    json_path = os.path.join(OUTPUT_DIR, f"job_{job_name}.json")
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Running Job: {job_name}")
    try:
        subprocess.check_call(["python3", RUN_JOB_SCRIPT, json_path])
    except subprocess.CalledProcessError as e:
        print(f"Error running job {job_name}: {e}")
        return False
    return True

def get_energies_for_photon_range(start_ph, end_ph, pts, ie):
    start_input = start_ph - ie
    end_input = end_ph - ie
    return np.linspace(start_input, end_input, pts).tolist()

def clean_dipole_str(d):
    s = f"{d:.6f}"
    s = s.rstrip('0').rstrip('.')
    if s == "": s = "0"
    return s

# --- Tasks ---

def plot_relative_channels():
    print("\n--- Plotting Relative XS (By Channel) ---")
    
    # Load existing CSVs
    all_data = []
    for d in DIPOLES_ALL:
        d_str = clean_dipole_str(d)
        fname = f"XS_Rel_pVTZ_D{d}.csv" # Clean name created by prev script
        path = os.path.join(OUTPUT_DIR, fname)
        
        # Fallback to potentially raw output if not renamed yet?
        # The interrupted script renamed them. If interrupted DURING renaming, some might be missing.
        # Check raw output name too?
        # Raw name: xs_relative_D{d_str}.csv
        
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_data.append(df)
        else:
            raw_path = os.path.join(OUTPUT_DIR, f"xs_relative_D{d_str}.csv")
            if os.path.exists(raw_path):
                 df = pd.read_csv(raw_path)
                 df["Dipole"] = d
                 all_data.append(df)
            else:
                print(f"Warning: Missing data for D={d}")

    if not all_data:
        print("No relative XS data found.")
        return

    df = pd.concat(all_data)
    df = df.sort_values(by=["Dipole", "E_photon"])
    
    plt.figure(figsize=(12, 8))
    
    channels = [c for c in df.columns if c.startswith("Rel_XS_Ch")]
    styles = ['-', '--', ':', '-.']
    # Colors: use cmap
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(DIPOLES_ALL), vmax=max(DIPOLES_ALL))
    
    for d in DIPOLES_ALL:
        subset = df[df["Dipole"] == d]
        if not subset.empty:
            color = cmap(norm(d))
            for i, ch in enumerate(channels):
                sty = styles[i % len(styles)]
                lbl = f"D={d} {ch}" if i == 0 else None # Only label 1st channel per dipole to avoid clutter? 
                # User asked for "for D... and for each channel". Legend will be huge.
                # Let's label fully.
                lbl = f"D={d} {ch.replace('Rel_XS_', '')}"
                plt.plot(subset["E_photon"], subset[ch], linestyle=sty, color=color, label=lbl, alpha=0.8)
            
    plt.xlabel("Photon Energy (eV)")
    plt.ylabel("Relative Cross Section (a.u.)")
    plt.title("Relative Cross Section: pVTZ b1 (By Channel)")
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Plot_Rel_pVTZ.png"), dpi=300)
    plt.close()
    print("Saved Plot_Rel_pVTZ.png")

def calc_and_plot_mid_range():
    print("\n--- Calculating & Plotting Mid Range pVTZ ---")
    
    # 1. Calc New Points (1.89 - 3.8 eV)
    energies_eke = get_energies_for_photon_range(1.89, 3.8, 50, IE_BASE)
    
    job_id = "Mid_pVTZ_Calc"
    job_config = {
        "qchem_output": os.path.abspath(PVTZ_OUT),
        "dyson": {
            "do_generation": True, 
            "indices": [0, 1],
            "grid_step": GRID_STEP,
            "padding": PADDING,
            "output_bin": os.path.join(os.path.abspath(OUTPUT_DIR), "dyson_pvtz_mid.bin")
        },
        "calculation": {
            "do_calculation": True,
            "skip_beta_gen": True,
            "type": "cross_section",
            "model": "point_dipole",
            "dipole_list": DIPOLES_ZOOM,
            "l_max": L_MAX,
            "points": 50,
            "ie": IE_BASE,
            "energies": energies_eke, 
            "output_csv": "dummy.csv"
        },
        "visualization": {"do_plot": False}
    }
    
    mid_data = []
    if run_job(job_config, job_id):
        for d in DIPOLES_ZOOM:
            d_str = clean_dipole_str(d)
            fname = f"cross_section_D{d_str}.txt"
            if os.path.exists(fname):
                data = []
                with open(fname, "r") as f:
                    for line in f:
                        if line.startswith("#"): continue
                        parts = line.split()
                        if len(parts) >= 2:
                            data.append({"E_photon": float(parts[0]), "CrossSection": float(parts[1]), "Dipole": d})
                # Rename/Save new data
                dest = os.path.join(OUTPUT_DIR, f"XS_Mid_pVTZ_D{d}.csv")
                df = pd.DataFrame(data)
                df.to_csv(dest, index=False)
                mid_data.append(df)
                os.remove(fname)
    
    # 2. Load Zoom Data
    zoom_data = []
    for d in DIPOLES_ZOOM:
        path = os.path.join(OUTPUT_DIR, f"XS_Zoom_pVTZ_D{d}.csv")
        if os.path.exists(path):
            zoom_data.append(pd.read_csv(path))
            
    # Combine
    if not mid_data and not zoom_data:
        print("No data for Mid Plot")
        return
        
    full_df = pd.concat(mid_data + zoom_data, ignore_index=True)
    full_df = full_df.sort_values(by=["Dipole", "E_photon"])
    
    # Plot
    plt.figure(figsize=(10, 6))
    for d in DIPOLES_ZOOM:
        subset = full_df[full_df["Dipole"] == d]
        if not subset.empty:
             plt.plot(subset["E_photon"], subset["CrossSection"], marker='o', markersize=3, label=f"D={d}")
             
    plt.xlabel("Photon Energy (eV)")
    plt.ylabel("Total Cross Section (a.u.)")
    plt.title("Total Cross Section: pVTZ b1 (1.779 - 3.8 eV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "Plot_Mid_pVTZ_b1.png"), dpi=300)
    plt.close()
    print("Saved Plot_Mid_pVTZ_b1.png")

if __name__ == "__main__":
    plot_relative_channels()
    calc_and_plot_mid_range()
