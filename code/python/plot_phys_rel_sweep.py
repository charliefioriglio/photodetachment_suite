import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
INPUT_DIR = "production_data/CuO cross sections/Physical_Dipole_Relative"
OUTPUT_DIR = INPUT_DIR  # Save plots in the same directory

def load_data(root_dir):
    all_files = glob.glob(os.path.join(root_dir, "**/*.csv"), recursive=True)
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

def plot_subset(df, title, filename, channels):
    plt.figure(figsize=(10, 6))
    
    # Get unique dipoles and lengths for styling
    dipoles = sorted(df["Dipole"].unique())
    lengths = sorted(df["DipoleLength"].unique())
    
    # Define styles
    # Color -> Dipole
    # Linestyle -> Channel
    # Marker -> Length (if multiple lengths in plot)
    
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis', len(dipoles))
    
    linestyles = ['-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D'] # For lengths if needed
    
    for i, d in enumerate(dipoles):
        for j, ch in enumerate(channels):
            # If multiple lengths, we iterate them too?
            if len(lengths) > 1:
                for k, length in enumerate(lengths):
                    subset = df[(df["Dipole"] == d) & (df["DipoleLength"] == length)]
                    if subset.empty: continue
                    subset = subset.sort_values("E_photon")
                    
                    label = f"D={d}, a={length}, {ch.replace('Rel_XS_', '')}"
                    plt.plot(subset["E_photon"], subset[ch], 
                             color=cmap(i), 
                             linestyle=linestyles[j % len(linestyles)],
                             marker=None, # Too crowded for markers usually
                             linewidth=1.5,
                             label=label)
            else:
                subset = df[df["Dipole"] == d]
                if subset.empty: continue
                subset = subset.sort_values("E_photon")
                
                label = f"D={d}, {ch.replace('Rel_XS_', '')}"
                plt.plot(subset["E_photon"], subset[ch], 
                         color=cmap(i), 
                         linestyle=linestyles[j % len(linestyles)],
                         linewidth=1.5,
                         label=label)

    plt.xlabel("Photon Energy (eV)")
    plt.ylabel("Relative Cross Section (Fraction)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    # Legend can be huge, put it outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"Saved {filename}")

def main():
    print("Loading data...")
    df = load_data(INPUT_DIR)
    if df.empty:
        print("No data found.")
        return
        
    channels = [c for c in df.columns if c.startswith("Rel_XS_Ch")]
    print(f"Found channels: {channels}")
    
    # Plot 1: a = 0.001
    print("Generating Plot 1 (a=0.001)...")
    df_near = df[df["DipoleLength"] == 0.001]
    if not df_near.empty:
        plot_subset(df_near, "Relative XS (Physical Dipole, a=0.001)", "Plot_Rel_Phys_a0.001.png", channels)
    
    # Plot 2: a = 1.578
    print("Generating Plot 2 (a=1.578)...")
    df_far = df[df["DipoleLength"] == 1.578]
    if not df_far.empty:
        plot_subset(df_far, "Relative XS (Physical Dipole, a=1.578)", "Plot_Rel_Phys_a1.578.png", channels)
        
    # Plot 3: All
    print("Generating Plot 3 (Combined)...")
    # For combined, maybe facet or just plot all? 
    # User asked for "one... for all dipole strengths and seperation distances".
    # I'll put them on one plot. styling might be tricky but I'll use the logic in plot_subset.
    plot_subset(df, "Relative XS (Physical Dipole, All Parameters)", "Plot_Rel_Phys_Combined.png", channels)

if __name__ == "__main__":
    main()
