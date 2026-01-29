
import os
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# Configuration
OUTPUT_DIR = "production_data"
RUN_JOB_SCRIPT = "src/python/run_job.py"
CONTINUUM_EXE = "./continuum_plotter"

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def msg(s):
    print(f"\n{'='*50}\n{s}\n{'='*50}\n")

# --- Task 1: Continuum Comparisons ---
def task_continuum():
    msg("Running Continuum Comparisons")
    out_dir = os.path.join(OUTPUT_DIR, "continuum")
    ensure_dir(out_dir)
    
    dipoles = [0.0, 0.1, 0.3, 0.5, 0.63]
    energies = [0.5] # Representative low energy
    l_max = 5
    
    # 1. Generate Data
    for D in dipoles:
        for E_ev in energies:
            E_au = E_ev / 27.211386
            
            # Radial (Now extended to r=100)
            rad_file = os.path.join(out_dir, f"rad_D{D}_E{E_ev}.csv")
            cmd = [CONTINUUM_EXE, "radial", str(D), str(E_au), str(l_max), rad_file]
            run_command(cmd)
            
            # Angular
            ang_file = os.path.join(out_dir, f"ang_D{D}_E{E_ev}.csv")
            cmd = [CONTINUUM_EXE, "angular", str(D), str(E_au), str(l_max), ang_file]
            run_command(cmd)
            
    # 2. Plotting
    for E_ev in energies:
        # Radial Plot (Standard r=20)
        plt.figure(figsize=(12, 8))
        for D in dipoles:
            df = pd.read_csv(os.path.join(out_dir, f"rad_D{D}_E{E_ev}.csv"))
            df_short = df[df['r'] <= 20.0]
            plt.plot(df_short['r'], df_short['Re_PD'], label=f"PD D={D}")
            
        df0 = pd.read_csv(os.path.join(out_dir, f"rad_D0.0_E{E_ev}.csv"))
        df0_short = df0[df0['r'] <= 20.0]
        plt.plot(df0_short['r'], df0_short['Re_PWE'], 'k--', label="PWE Ref")
        plt.plot(df0_short['r'], df0_short['Re_Analytic'], 'g:', label="Analytic PW")
        
        plt.title(f"Radial Wavefunction (Re) along Z-axis @ {E_ev} eV")
        plt.xlabel("r (Bohr)")
        plt.ylabel("Re(Psi)")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "radial_comparison.png"))
        plt.close()

        # Radial Plot (Extended r=100)
        plt.figure(figsize=(12, 8))
        for D in dipoles:
            df = pd.read_csv(os.path.join(out_dir, f"rad_D{D}_E{E_ev}.csv"))
            plt.plot(df['r'], df['Re_PD'], label=f"PD D={D}")
            
        df0 = pd.read_csv(os.path.join(out_dir, f"rad_D0.0_E{E_ev}.csv"))
        plt.plot(df0['r'], df0['Re_PWE'], 'k--', label="PWE Ref")
        # plt.plot(df0['r'], df0['Re_Analytic'], 'g:', label="Analytic PW") # Optional
        
        plt.title(f"Radial Wavefunction (Re) along Z-axis (Extended) @ {E_ev} eV")
        plt.xlabel("r (Bohr)")
        plt.ylabel("Re(Psi)")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "radial_comparison_extended.png"))
        plt.close()
        
        # Angular Plot
        plt.figure(figsize=(12, 8))
        for D in dipoles:
            df = pd.read_csv(os.path.join(out_dir, f"ang_D{D}_E{E_ev}.csv"))
            mag_sq = df['Re_PD']**2 + df['Im_PD']**2
            plt.plot(df['theta'], mag_sq, label=f"PD D={D}")
            
        df0 = pd.read_csv(os.path.join(out_dir, f"ang_D0.0_E{E_ev}.csv"))
        mag_sq_pwe = df0['Re_PWE']**2 + df0['Im_PWE']**2
        plt.plot(df0['theta'], mag_sq_pwe, 'k--', label="PWE Ref")
        
        plt.title(f"Angular Distribution |Psi|^2 @ {E_ev} eV (r=10)")
        plt.xlabel("Theta (deg)")
        plt.ylabel("|Psi|^2")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "angular_comparison.png"))
        plt.close()


# --- Task 2: CN- Beta ---
def task_cn():
    msg("Running CN- Beta Calculations")
    out_dir = os.path.join(OUTPUT_DIR, "CN")
    ensure_dir(out_dir)
    
    dipoles = [0.0, 0.1, 0.3, 0.5, 0.57, 0.63]
    energies = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    overall_results = []
    
    for D in dipoles:
        job_file = os.path.join(out_dir, f"job_D{D}.json")
        res_file = os.path.join(out_dir, f"beta_D{D}.csv")
        
        config = {
            "qchem_output": "reference materials/CN.out",
            "dyson": {
                "do_generation": True,
                "indices": [0, 1], # Correct indices for Left/Right
                "grid_step": 0.4,
                "padding": 20.0,
                "output_bin": os.path.join(out_dir, "cn_dyson.bin")
            },
            "calculation": {
                "do_calculation": True,
                "type": "beta",
                "model": "point_dipole",
                "dipole": D,
                "l_max": 5,
                "points": 150, # Use Lebedev Hardcoded
                "ie": 3.977,
                "energies": energies,
                "output_csv": res_file
            },
            "visualization": {"do_plot": False}
        }
        
        if D != dipoles[0]: config["dyson"]["do_generation"] = False
        
        with open(job_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        run_command(["python3", RUN_JOB_SCRIPT, job_file])
        
        if os.path.exists(res_file):
            df = pd.read_csv(res_file)
            df['Dipole'] = D
            overall_results.append(df)
            
    # Plotting
    if overall_results:
        full_df = pd.concat(overall_results)
        full_df.to_csv(os.path.join(out_dir, "cn_all_results.csv"), index=False)
        
        # Beta Plot (Linear Scale)
        plt.figure(figsize=(10, 6))
        for D in dipoles:
            subset = full_df[full_df['Dipole'] == D]
            plt.plot(subset['eKE'], subset['Beta'], 'o-', label=f"D={D}")
        
        plt.xlabel("eKE (eV)")
        plt.ylabel("Beta")
        plt.title("CN- Beta Parameter vs Energy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "cn_beta_plot.png"))
        plt.close()

# --- Task 4: CuO Relative Cross Sections (Vibrational) ---
def task_cuo_relative():
    print("\n==================================================")
    print("Running CuO Relative Cross Sections (Vib)")
    print("==================================================")
    
    out_dir = os.path.join(OUTPUT_DIR, "CuO_Relative_Vib")
    ensure_dir(out_dir)
    
    # 1. Create Vibration Data File
    # Format: E_bind FCF (Overlap)
    # User provided: 
    # 1.7780 0.8492010
    # 1.8574 0.5006826
    # 1.9367 0.2056282
    vib_file_path = os.path.join(out_dir, "vib_data.txt")
    with open(vib_file_path, "w") as f:
        f.write("1.7780 0.8492010\n")
        f.write("1.8574 0.5006826\n")
        f.write("1.9367 0.2056282\n")
        
    # Photon Energies (from 1.8 to 3.0)
    # Using dense grid for smooth plots
    e_ph_grid = np.linspace(1.75, 3.0, 126).tolist()
    
    configs = [
        {"label": "pVTZ", "file": "reference materials/CuO_pVTZ.out", "indices": [0, 1]},
        {"label": "Aug",  "file": "reference materials/CuO_augccPVDZPP_dyson.out", "indices": [2, 3]}
    ]
    
    dipoles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    for cfg in configs:
        label = cfg["label"]
        qchem_file = os.path.join(os.getcwd(), cfg["file"]) # Ensure absolute or relative corrrect
        
        all_results = []
        
        for i, D in enumerate(dipoles):
            job_name = f"xs_{label}_D{D}"
            json_path = os.path.join(out_dir, f"job_{job_name}.json")
            
            # Note: For efficiency, we could generate Dyson once per file.
            # But run_job handles generation. 
            # We can re-use the bin file if we skip generation.
            # But the 'vib_file' arg is passed to dyson_io during GENERATION step (or pseudo-generation).
            # Actually, my dyson_io logic runs the C++ tool.
            # So I must run "generation" step every time if I want to trigger the calculation via dyson_io.
            # (Since dyson_io calls the binary).
            # So do_generation=True is required to invoke dyson_io and thus the binary to compute XS.
            
            job_data = {
                "qchem_output": qchem_file,
                "dyson": {
                    "do_generation": True,
                    "indices": cfg["indices"],
                    "grid_step": 0.4, 
                    "padding": 20.0,
                    "vib_file": vib_file_path, # Triggers Relative XS output in C++
                    "output_bin": f"{out_dir}/{label}_dyson.bin"
                },
                "calculation": {
                    "do_calculation": True, # Required to pass XS params to dyson_io
                    "skip_beta_gen": True,
                    "type": "cross_section",
                    "model": "point_dipole",
                    # Actually run_job WILL run beta_gen later if do_calculation is True.
                    # We can ignore beta_gen output or let it fail/overwrite.
                    # Ideally we disable beta_gen run? 
                    # If I set do_calculation=False, dyson_io won't get XS params!
                    # Catch-22 in run_job.py logic.
                    # I will let it run beta_gen. It might just overwrite 'output_csv'.
                    # But beta_gen takes 'energies' as E_ph.
                    # If beta_gen runs, it calculates Beta?
                    # I'll set output_csv to a dummy.
                    "dipole": D,
                    "l_max": 4,
                    "points": 100, 
                    "ie": 1.778, # Dummy
                    "energies": e_ph_grid, 
                    "output_csv": f"{out_dir}/{job_name}_beta_dummy.csv" 
                },
                "visualization": { "do_plot": False }
            }
            
            with open(json_path, "w") as f:
                json.dump(job_data, f, indent=2)
                
            run_command(["python3", RUN_JOB_SCRIPT, json_path])
            
            # Collect xs_relative.csv
            # It is written to CWD by dyson_io/dyson_gen execution.
            rel_csv = "xs_relative.csv" 
            target_csv = os.path.join(out_dir, f"rel_{label}_D{D}.csv")
            
            if os.path.exists(rel_csv):
                os.rename(rel_csv, target_csv)
                df = pd.read_csv(target_csv)
                df["Dipole"] = D
                all_results.append(df)
            else:
                print(f"Warning: {rel_csv} not found for {job_name}")
                
        # Stitch and Plot
        if all_results:
            df_final = pd.concat(all_results)
            df_final.to_csv(os.path.join(out_dir, f"all_relative_{label}.csv"), index=False)
            
            # Plot
            plt.figure(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(dipoles)))
            linestyles = ['-', '--', ':'] # for v=0, 1, 2
            
            # Check how many channels in CSV
            # Format: E_photon, Total_XS, Rel_XS_Ch0, Rel_XS_Ch1...
            # We want Rel_XS_ChX
            
            cols = [c for c in df_final.columns if "Rel_XS_Ch" in c]
            
            for i, D in enumerate(dipoles):
                subset = df_final[df_final["Dipole"] == D]
                for v_idx, col in enumerate(cols):
                    if v_idx >= len(linestyles): ls = '-'
                    else: ls = linestyles[v_idx]
                    
                    plt.plot(subset["E_photon"], subset[col], 
                             color=colors[i], linestyle=ls, 
                             label=f"D={D} v={v_idx}" if i==0 or i==len(dipoles)-1 else "")
                             
            plt.xlabel("Photon Energy (eV)")
            plt.ylabel("Relative Cross Section")
            plt.title(f"Relative XS Evolution: {label}")
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, f"plot_relative_xs_{label}.png"))
            plt.close()



def main():
    ensure_dir(OUTPUT_DIR)
    # task_continuum() 
    
    # Priority Tasks:
    # task_cn()          # Done
    task_cuo_relative() # NEW Task
    
    # task_cuo()       # Skip old task
    print("Done generating production data.")


if __name__ == "__main__":
    main()
