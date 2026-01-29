
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

# Files
PVTZ_OUT = "reference materials/CuO_pVTZ.out"
AUG_OUT = "reference materials/CuO_augccPVDZPP_dyson.out"

# Common Params
IE_BASE = 1.778
L_MAX = 4
GRID_STEP = 0.3
PADDING = 20.0
POINTS = 100 

# Dipoles
DIPOLES_ALL = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
DIPOLES_ZOOM = [0.0, 0.1, 0.2, 0.4, 0.6]

# --- Helpers ---
def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

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

def standardize_energies_eke(start_eke, end_eke, pts):
    return np.linspace(start_eke, end_eke, pts).tolist()

def get_energies_for_photon_range(start_ph, end_ph, pts, ie):
    start_input = start_ph - ie
    end_input = end_ph - ie
    return np.linspace(start_input, end_input, pts).tolist()

def clean_dipole_str(d):
    # Matches C++ output format logic exactly
    s = f"{d:.6f}"
    s = s.rstrip('0').rstrip('.')
    if s == "": s = "0"
    return s

# --- Tasks ---

def task_broad_range():
    print("\n--- Task 1: Broad Range (0.1 - 10.1 eKE) ---")
    
    tasks = [
        {"name": "pVTZ_b1", "file": PVTZ_OUT, "indices": [0, 1]},
        {"name": "Aug_b1", "file": AUG_OUT, "indices": [2, 3]},
        {"name": "Aug_a1", "file": AUG_OUT, "indices": [0, 1]},
    ]
    
    energies_eke = standardize_energies_eke(0.1, 10.1, 50)
    results_map = {} 
    
    for t in tasks:
        t_name = t["name"]
        t_file = t["file"]
        t_idx = t["indices"]
        
        job_id = f"Broad_{t_name}"
        
        job_config = {
            "qchem_output": os.path.abspath(t_file),
            "dyson": {
                "do_generation": True,
                "indices": t_idx,
                "grid_step": GRID_STEP,
                "padding": PADDING,
                "output_bin": os.path.join(os.path.abspath(OUTPUT_DIR), f"dyson_{t_name}.bin")
            },
            "calculation": {
                "do_calculation": True,
                "skip_beta_gen": True,
                "type": "cross_section",
                "model": "point_dipole",
                "dipole_list": DIPOLES_ALL, # Optimized List Call
                "l_max": L_MAX,
                "points": 50,
                "ie": IE_BASE,
                "energies": energies_eke,
                "output_csv": "dummy.csv"
            },
            "visualization": {"do_plot": False}
        }
        
        if run_job(job_config, job_id):
            # Collect outputs
            all_df_t = []
            for d in DIPOLES_ALL:
                d_str = clean_dipole_str(d)
                fname = f"cross_section_D{d_str}.txt"
                if os.path.exists(fname):
                    # Rename to safe place
                    csv_name = f"XS_Broad_{t_name}_D{d}.csv"
                    dest = os.path.join(OUTPUT_DIR, csv_name)
                    
                    data = []
                    with open(fname, "r") as f:
                        for line in f:
                            if line.startswith("#"): continue
                            parts = line.split()
                            if len(parts) >= 2:
                                e_ph = float(parts[0])
                                sigma = float(parts[1])
                                data.append({"E_photon": e_ph, "eKE": e_ph - IE_BASE, "CrossSection": sigma, "Dipole": d})
                    
                    df = pd.DataFrame(data)
                    df.to_csv(dest, index=False)
                    all_df_t.append(df)
                    os.remove(fname) # Clean up
            
            if all_df_t:
                full_df = pd.concat(all_df_t)
                results_map[t_name] = full_df

    # Plotting Broad
    for t_name, df in results_map.items():
        plt.figure(figsize=(10, 6))
        for d in DIPOLES_ALL:
            subset = df[df["Dipole"] == d]
            if not subset.empty:
                plt.plot(subset["eKE"], subset["CrossSection"], marker='o', markersize=3, label=f"D={d}")
                
        plt.xlabel("Electron Kinetic Energy (eV)")
        plt.ylabel("Total Cross Section (a.u.)")
        plt.title(f"Total Cross Section: {t_name} (Broad Range)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f"Plot_Broad_{t_name}.png"), dpi=300)
        plt.close()
        print(f"Saved Plot_Broad_{t_name}.png")

def task_zoomed_pvtz():
    print("\n--- Task 2: Zoomed pVTZ (1.779 - 1.88 eV Eph) ---")
    
    energies_eke = get_energies_for_photon_range(1.779, 1.88, 50, IE_BASE)
    
    job_id = "Zoom_pVTZ"
    job_config = {
        "qchem_output": os.path.abspath(PVTZ_OUT),
        "dyson": {
            "do_generation": True, 
            "indices": [0, 1],
            "grid_step": GRID_STEP,
            "padding": PADDING,
            "output_bin": os.path.join(os.path.abspath(OUTPUT_DIR), "dyson_pvtz_zoom.bin")
        },
        "calculation": {
            "do_calculation": True,
            "skip_beta_gen": True,
            "type": "cross_section",
            "model": "point_dipole",
            "dipole_list": DIPOLES_ZOOM, # Zoom List
            "l_max": L_MAX,
            "points": 50,
            "ie": IE_BASE,
            "energies": energies_eke, 
            "output_csv": "dummy.csv"
        },
        "visualization": {"do_plot": False}
    }
    
    all_data = []
    if run_job(job_config, job_id):
        for d in DIPOLES_ZOOM:
            d_str = clean_dipole_str(d)
            fname = f"cross_section_D{d_str}.txt"
            if os.path.exists(fname):
                csv_name = f"XS_Zoom_pVTZ_D{d}.csv"
                dest = os.path.join(OUTPUT_DIR, csv_name)
                
                data = []
                with open(fname, "r") as f:
                    for line in f:
                        if line.startswith("#"): continue
                        parts = line.split()
                        if len(parts) >= 2:
                            data.append({"E_photon": float(parts[0]), "CrossSection": float(parts[1]), "Dipole": d})
                df = pd.DataFrame(data)
                df.to_csv(dest, index=False)
                all_data.append(df)
                os.remove(fname)

    if all_data:
        df = pd.concat(all_data)
        plt.figure(figsize=(10, 6))
        for d in DIPOLES_ZOOM:
            subset = df[df["Dipole"] == d]
            if not subset.empty:
                plt.plot(subset["E_photon"], subset["CrossSection"], marker='o', markersize=3, label=f"D={d}")
        
        plt.xlabel("Photon Energy (eV)")
        plt.ylabel("Total Cross Section (a.u.)")
        plt.title("Total Cross Section: pVTZ b1 (Zoomed Region)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "Plot_Zoom_pVTZ.png"), dpi=300)
        plt.close()
        print("Saved Plot_Zoom_pVTZ.png")

def task_relative_pvtz():
    print("\n--- Task 3: Relative XS pVTZ (1.78 - 2.3 eV) ---")
    
    vib_file = os.path.join(OUTPUT_DIR, "vib_data_pvtz.txt")
    with open(vib_file, "w") as f:
        f.write("1.7780 0.8492010\n")
        f.write("1.8574 0.5006826\n")
        f.write("1.9367 0.2056282\n")
        
    energies = np.linspace(1.78, 2.3, 100).tolist()
    energies = [round(e, 4) for e in energies] 
    
    job_id = "Rel_pVTZ"
    job_config = {
        "qchem_output": os.path.abspath(PVTZ_OUT),
        "dyson": {
            "do_generation": True,
            "indices": [0, 1],
            "grid_step": GRID_STEP,
            "padding": PADDING,
            "vib_file": os.path.abspath(vib_file),
            "output_bin": os.path.join(os.path.abspath(OUTPUT_DIR), "dyson_pvtz_rel.bin")
        },
        "calculation": {
            "do_calculation": True,
            "skip_beta_gen": True,
            "type": "cross_section",
            "model": "point_dipole",
            "dipole_list": DIPOLES_ALL,
            "l_max": L_MAX,
            "points": 100,
            "ie": IE_BASE,
            "energies": energies, 
            "output_csv": "dummy.csv"
        },
        "visualization": {"do_plot": False}
    }
    
    all_data = []
    if run_job(job_config, job_id):
        for d in DIPOLES_ALL:
            d_str = clean_dipole_str(d)
            fname = f"xs_relative_D{d_str}.csv"
            if os.path.exists(fname):
                csv_name = f"XS_Rel_pVTZ_D{d}.csv"
                dest = os.path.join(OUTPUT_DIR, csv_name)
                
                df = pd.read_csv(fname)
                df["Dipole"] = d
                df.to_csv(dest, index=False)
                all_data.append(df)
                os.remove(fname)
                

    # Plotting Relative
    if all_data:
        df = pd.concat(all_data)
        # Sort to ensure lines connect properly
        df = df.sort_values(by=["Dipole", "E_photon"])
        
        plt.figure(figsize=(10, 6))
        
        # Determine channels (columns starting with Rel_XS_Ch)
        channels = [c for c in df.columns if c.startswith("Rel_XS_Ch")]
        
        # Colors for channels
        # Strategy: Dipole defines color/marker, Channel defines linestyle? 
        # Or just plot everything. User said "for D in dipole and for each vibrational channel".
        # Let's align colors with dipoles, and linestyles with channels.
        
        styles = ['-', '--', ':', '-.']
        
        for d in DIPOLES_ALL:
            subset = df[df["Dipole"] == d]
            if not subset.empty:
                # Plot Total
                # plt.plot(subset["E_photon"], subset["Total_XS"], label=f"D={d} Total", alpha=0.5, linewidth=1)
                
                for i, ch in enumerate(channels):
                    sty = styles[i % len(styles)]
                    lbl = f"D={d} {ch}"
                    plt.plot(subset["E_photon"], subset[ch], linestyle=sty, label=lbl, alpha=0.8)
                
        plt.xlabel("Photon Energy (eV)")
        plt.ylabel("Relative Cross Section (a.u.)")
        plt.title("Relative Cross Section: pVTZ b1 (By Channel)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "Plot_Rel_pVTZ.png"), dpi=300)
        plt.close()
        print("Saved Plot_Rel_pVTZ.png")

def task_mid_pvtz():
    print("\n--- Task 4: Mid Range pVTZ (1.779 - 3.779 eV Eph) ---")
    
    # 1. New Calculation: 1.89 to 3.8 eV (50 pts)
    # Note: User said 3.8, but range title says 3.779. I'll stick to 1.89 to 3.8 as requested explicitly for points.
    energies_eke = get_energies_for_photon_range(1.89, 3.8, 50, IE_BASE)
    
    job_id = "Mid_pVTZ"
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
            "dipole_list": DIPOLES_ZOOM, # Using Zoom dipoles for consistency
            "l_max": L_MAX,
            "points": 50,
            "ie": IE_BASE,
            "energies": energies_eke, 
            "output_csv": "dummy.csv"
        },
        "visualization": {"do_plot": False}
    }
    
    # Run newly requested points
    mid_data = [] # List of DFs
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
                df = pd.DataFrame(data)
                mid_data.append(df)
                os.remove(fname)

    # 2. Convert collected mid_data to single DF
    if mid_data:
        df_mid = pd.concat(mid_data)
    else:
        df_mid = pd.DataFrame()
        
    # 3. Load Zoom Data
    zoom_data = []
    for d in DIPOLES_ZOOM:
        csv_name = f"XS_Zoom_pVTZ_D{d}.csv"
        path = os.path.join(OUTPUT_DIR, csv_name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            zoom_data.append(df)
            
    if zoom_data:
        df_zoom = pd.concat(zoom_data)
    else:
        df_zoom = pd.DataFrame()
        
    # 4. Combine
    if df_zoom.empty and df_mid.empty:
        print("No data for Mid Plot")
        return

    full_df = pd.concat([df_zoom, df_mid], ignore_index=True)
    full_df = full_df.sort_values(by=["Dipole", "E_photon"])
    
    # Save Combined Data? User didn't ask, but good practice.
    # full_df.to_csv(os.path.join(OUTPUT_DIR, "XS_Mid_pVTZ_Combined.csv"), index=False)
    
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

def main():
    ensure_dir(OUTPUT_DIR)
    
    # task_broad_range()
    # task_zoomed_pvtz()
    task_relative_pvtz() # Regenerates Plot only (if logic updated)
    task_mid_pvtz()
    
    print("\nProduction Run Complete.")
    print(f"Outputs in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
