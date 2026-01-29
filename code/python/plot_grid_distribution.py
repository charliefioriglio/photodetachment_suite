import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def read_grid(filename):
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    data.append([float(p) for p in parts[:3]]) # Alpha, Beta, Gamma
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return np.array([])
    return np.array(data)

def plot_grid(ax, data, title):
    if len(data) == 0: return
    alpha = data[:, 0]
    beta = data[:, 1]
    
    # Convert to Cartesian on Unit Sphere
    # User's ref code:
    # x = sin(theta) * cos(phi)
    # y = sin(theta) * sin(phi)
    # z = cos(theta)
    # Here theta=beta, phi=alpha
    
    x = np.sin(beta) * np.cos(alpha)
    y = np.sin(beta) * np.sin(alpha)
    z = np.cos(beta)
    
    ax.scatter(x, y, z, s=10, alpha=0.8)
    ax.set_title(f"{title} ({len(data)} pts)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Draw sphere wireframe for reference
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = np.cos(u)*np.sin(v)
    ys = np.sin(u)*np.sin(v)
    zs = np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.1)

def main():
    files = [
        ("grid_hardcoded.txt", "Hardcoded (150)"),
        ("grid_geom.txt", "Geometric (50)"),
        ("grid_rep.txt", "Repulsion (50)")
    ]
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, (fname, title) in enumerate(files):
        data = read_grid(fname)
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        plot_grid(ax, data, title)
        
    plt.tight_layout()
    plt.savefig("angle_plots.png", dpi=150)
    print("Saved angle_plots.png")

if __name__ == "__main__":
    main()
