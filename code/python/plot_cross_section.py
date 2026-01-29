import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_xs(file, output):
    # Read CSV
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return

    # Columns: E_photon(eV), Total_XS, Ch0_..., Ch1_...
    E_ph = df.iloc[:, 0]
    Total_XS = df.iloc[:, 1]
    
    plt.figure(figsize=(10, 6))
    
    # Plot Total
    plt.plot(E_ph, Total_XS, 'k-', linewidth=2, label='Total Cross Section')
    
    # Plot Channels
    # Channels start from index 2
    for col in df.columns[2:]:
        # Parse legend label "Ch0_Ebind=1.778" -> "v=0 (1.778 eV)"
        label = col
        if "Ch" in col and "Ebind" in col:
            parts = col.split('_')
            ch_idx = parts[0].replace("Ch", "")
            ebind = parts[1].replace("Ebind=", "")
            label = f"v={ch_idx} ({ebind} eV)"
            
        plt.plot(E_ph, df[col], '--', linewidth=1.5, alpha=0.8, label=label)
        
    plt.xlabel("Photon Energy (eV)", fontsize=12)
    plt.ylabel("Cross Section (a.u.)", fontsize=12)
    plt.title("Photodetachment Cross Section", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print(f"Saving plot to {output}...")
    plt.savefig(output, dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Cross Sections")
    parser.add_argument("file", help="Input CSV file (xs_relative.csv)")
    parser.add_argument("--output", default="cross_section_plot.png", help="Output image file")
    
    args = parser.parse_args()
    
    plot_xs(args.file, args.output)
