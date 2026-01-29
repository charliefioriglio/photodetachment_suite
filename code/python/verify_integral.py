import numpy as np
import argparse
from dyson_io import load_qchem, write_cpp_input
from scipy.special import spherical_jn, sph_harm

def planewave_expansion(k, l, m, x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    # Clip z/r to avoid domain errors
    theta = np.arccos(np.clip(z / (r + 1e-12), -1.0, 1.0))
    phi = np.arctan2(y, x)
    R = spherical_jn(l, k * r)
    Y = sph_harm(m, l, phi, theta) # Note: scipy uses (m, l, phi, theta) -> (m, n, theta, phi) in docs? 
    # scipy.special.sph_harm(m, n, theta, phi). theta is azimuthal (0..2pi), phi is polar (0..pi).
    # Wait. scipy 1.9+: sph_harm(m, n, theta, phi). theta=azimuthal(lon), phi=polar(colat).
    # Wikipedia/Physics convention: theta=polar, phi=azimuthal.
    # My C++: theta=polar (0..pi), phi=azimuthal (0..2pi).
    # Scipy docs: "theta: azimuthal coordinate (0, 2pi), phi: polar coordinate (0, pi)".
    # So scipy(m, l, phi_ang, theta_pol).
    return Y * R

def integrate_3d(f, dV):
    return np.sum(f) * dV

def main(input_file, i1, i2, ie_ev, e_ph_ev, l_max=0):
    data = load_qchem(input_file)
    print(f"Loaded {len(data.atoms)} atoms, {len(data.dyson_orbitals)} DOs")
    
    # Grid parameters (match fine grid)
    padding = 5.0
    step = 0.2
    
    coords = np.array([a.center_bohr for a in data.atoms])
    min_c = coords.min(axis=0) - padding
    max_c = coords.max(axis=0) + padding
    
    x = np.arange(min_c[0], max_c[0] + 1e-9, step)
    y = np.arange(min_c[1], max_c[1] + 1e-9, step)
    z = np.arange(min_c[2], max_c[2] + 1e-9, step)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    dV = step**3
    print(f"Grid shape: {X.shape}, dV={dV}")
    
    # Build Dyson Orbitals
    # Need to reimplement 'build_DO' logic or reuse it?
    # dyson_io.py doesn't have 'evaluate' on grid.
    # But C++ generated 'cuo_b1_opt.bin'. I could read that?
    # Better to calculate fresh to verify C++ 'evaluate' too?
    # Implementation of Gaussian eval in python is tedious.
    # Let's read the binary for now?
    # OR: use the C++ binary output if we trust 'evaluate'.
    # If the issue is math_special (bessel/Ylm), we can check that.
    
    # Let's assume DO evaluation is correct (validated by norm).
    # We will use the C++ 'evaluate' output from binary.
    # But binary stores only ONE orbital (L?).
    # My dyson_gen writes L, but not R if combined?
    # dyson_gen writes 'data' which is primary_dyson (L).
    # So we can check L integral.
    
    # Read binary
    # Format: NX NY NZ X0 Y0 Z0 DX DY DZ [DATA...]
    # dyson_io doesn't strictly follow this format in 'write_binary'?
    # C++ 'write_binary' writes strictly data?
    # grid.write_binary: "Format: NX NY NZ..." ?
    # Let's check grid.cpp.
    
    # Actually, simpler to just implement S-wave integral using scipy bessel/Ylm and compare with C++ logic?
    # I want to check the *magnitude*.
    pass

if __name__ == "__main__":
    main("reference materials/CuO_augccPVDZPP_dyson.out", 2, 3, 1.778, 1.88)
