import pandas as pd
import sys
import os

def load_xs(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def main():
    root = "xs_test"
    files = {
        "PWE": "pwe.csv",
        "Point Dipole (D=0)": "pd.csv",
        "Physical Dipole (D=0, a=0.01)": "phys.csv"
    }
    
    results = {}
    
    print("--- Cross Section Comparison ---")
    for label, fname in files.items():
        path = os.path.join(root, fname)
        df = load_xs(path)
        if df is not None:
            # Get XS at the target energy (approx 0.508 eV eKE -> 2.286 eV Photon)
            # Actually pwe.csv has multiple energies. 
            # pd/phys might have 1 or 3.
            # Let's print all rows
            print(f"\n[{label}]")
            print(df.to_string(index=False))
            
            results[label] = df
            
    # Check consistency at ~2.286 eV (Row 2 in PWE?)
    # PWE: 1.878, 2.08208, 2.28616
    # PD: 0.1, 0.3, 0.508 (eKE) -> +1.778 = 1.878, 2.082, 2.286
    pass

if __name__ == "__main__":
    main()
