import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot Beta Parameter vs eKE")
    parser.add_argument("input_csv", help="Path to input CSV file with beta data")
    parser.add_argument("--output", default="beta_plot.png", help="Output PNG filename (default: beta_plot.png)")
    parser.add_argument("--show", action="store_true", help="Display plot window")
    parser.add_argument("--title", default="Photoelectron Angular Distribution Anisotropy", help="Plot title")
    args = parser.parse_args()
    
    beta_file = args.input_csv
        
    if not os.path.exists(beta_file):
        print(f"Error: {beta_file} not found.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(beta_file)
    except Exception as e:
        print(f"Error reading {beta_file}: {e}")
        sys.exit(1)
        
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df["eKE"], df["Beta"], marker='o', linestyle='-', linewidth=2, label="Calculated Beta")
    
    plt.xlabel("Photoelectron KE (eV)")
    plt.ylabel(r"Anisotropy Parameter $\beta$")
    plt.title(args.title)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.ylim(-1.5, 2.5)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend()
    
    output_png = args.output
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_png}")
    
    if args.show:
        plt.show()
    
    # Print table
    print("\nResults:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
