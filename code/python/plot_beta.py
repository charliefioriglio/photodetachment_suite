import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def main():
    beta_file = "beta.csv"
    if len(sys.argv) > 1:
        beta_file = sys.argv[1]
        
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
    plt.title("Photoelectron Angular Distribution Anisotropy")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.ylim(-1.5, 2.5)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend()
    
    output_png = "beta_plot.png"
    plt.savefig(output_png, dpi=150)
    print(f"Saved plot to {output_png}")
    
    # Print table
    print("\nResults:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
