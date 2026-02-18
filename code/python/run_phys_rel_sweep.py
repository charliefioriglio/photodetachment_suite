import os
import json
import shutil
import subprocess
import glob
import numpy as np
import pandas as pd

# Configuration Parameters matching gen_cuo_final_xs.py
PVTZ_OUT = os.path.abspath("reference materials/CuO_pVTZ.out")
OUTPUT_DIR = os.path.abspath("production_data/CuO cross sections/Physical_Dipole_Relative")
VIB_FILE = os.path.join(OUTPUT_DIR, "vib_data_pvtz.txt")

# Ensure Output Directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Write Vibrational Data (Exact copy from snippet)
with open(VIB_FILE, "w") as f:
    f.write("1.7780 0.8492010\n")
    f.write("1.8574 0.5006826\n")
    f.write("1.9367 0.2056282\n")

# Energy Grid (Exact match to snippet)
energies = np.linspace(1.78, 2.3, 100).tolist()
energies = [round(e, 4) for e in energies]

# Sweep Parameters
dipoles = [0.0, 0.5, 1.0, 1.78]
lengths = [0.001, 1.578] # "a" values

# Dyson Parameters
# Snippet used indices [0, 1]
dyson_indices = [0, 1]
grid_step = 0.3 # Matching snippet GRID_STEP assumption (standard is 0.3)
padding = 20.0  # Matching snippet PADDING assumption

def clean_dipole_str(d):
    """Matches C++ to_string formatting for filename matching."""
    s = f"{d:.6f}".rstrip('0').rstrip('.')
    if s == "": s = "0"
    if d == 0.0: s = "0" # Special handle if needed, but C++ uses %g-like
    # Actually dyson_gen.cpp: 
    # d_str.erase(d_str.find_last_not_of('0') + 1, std::string::npos);
    # if(d_str.back() == '.') d_str.pop_back();
    # This logic results in "0" for 0.0, "0.5" for 0.5.
    return s

# Main Loop
for a_val in lengths:
    print(f"\n--- Processing a = {a_val} ---")
    
    # Subdirectory for this 'a' value
    sub_dir = os.path.join(OUTPUT_DIR, f"a_{a_val}")
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    job_config = {
        "qchem_output": PVTZ_OUT,
        "dyson": {
            "do_generation": True,
            "indices": dyson_indices,
            "grid_step": grid_step,
            "padding": padding,
            "vib_file": VIB_FILE,
            "output_bin": os.path.join(OUTPUT_DIR, f"dyson_pvtz_rel_a{a_val}.bin")
        },
        "calculation": {
            "do_calculation": True,
            "skip_beta_gen": True, # Relative calculation is done in dyson_gen
            "type": "cross_section",
            "model": "physical_dipole",
            "dipole_list": dipoles,
            "dipole_length": a_val,
            "l_max": 4, # Standard
            "ie": 1.778,
            "energies": energies,
            "points": 1
        },
        "visualization": {"do_plot": False}
    }
    
    # Write JSON
    json_name = f"job_phys_rel_a{a_val}.json"
    with open(json_name, "w") as f:
        json.dump(job_config, f, indent=2)
        
    # Run
    try:
        subprocess.check_call(["python3", "src/python/run_job.py", json_name])
    except subprocess.CalledProcessError as e:
        print(f"Error running job for a={a_val}: {e}")
        continue
        
    # Move and Rename Results
    for d in dipoles:
        d_str_cpp = clean_dipole_str(d)
        src_csv = f"xs_relative_D{d_str_cpp}.csv"
        
        if os.path.exists(src_csv):
            # Format: XS_Rel_Phys_a{a}_D{d}.csv
            dest_name = f"XS_Rel_Phys_a{a_val}_D{d}.csv"
            dest_path = os.path.join(sub_dir, dest_name)
            
            # Read and Add Metadata (like snippet)
            df = pd.read_csv(src_csv)
            df["Dipole"] = d
            df["DipoleLength"] = a_val
            df.to_csv(dest_path, index=False)
            
            print(f"Saved {dest_path}")
            os.remove(src_csv)
        else:
            print(f"Warning: Output {src_csv} not found.")

    # Cleanup JSON
    if os.path.exists(json_name):
        os.remove(json_name)

print("\nBatch Completed.")
