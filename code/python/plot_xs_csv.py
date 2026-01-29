import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_xs_csv.py input.csv output.png [Title]")
        sys.exit(1)
        
    input_csv = sys.argv[1]
    output_png = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "Cross Section"
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
        
    # Expect columns: "Energy", "CrossSection"
    # Energy is Photon Energy (eV).
    # Calculate eKE assuming IE=1.778 for CuO
    ie = 1.778
    if "Energy" in df.columns:
        df["eKE"] = df["Energy"] - ie
    
    plt.figure(figsize=(8, 6))
    plt.plot(df["Energy"], df["CrossSection"], 'o-', label="Cross Section")
    
    plt.xlabel("Photon Energy (eV)")
    plt.ylabel("Cross Section (a.u.)")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add eKE secondary axis? Or just mention
    # plt.twiny?
    # Let's keep it simple for verified output.
    
    print(f"Saving plot to {output_png}...")
    plt.savefig(output_png)
    plt.close()

if __name__ == "__main__":
    main()
