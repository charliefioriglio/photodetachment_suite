import numpy as np

def integrate_3d(f, dV):
    return np.sum(f) * dV

# ---------------------------------------------
# Double factorial
# ---------------------------------------------
def double_factorial(n):
    if n <= 0:
        return 1
    else:
        return n * double_factorial(n - 2)

# ---------------------------------------------
# Normalization factor for Cartesian Gaussians
# ---------------------------------------------
def norm_cartesian_gaussian(alpha, a, b, c):
    l = a + b + c
    prefactor = (2 * alpha / np.pi)**(3/4)
    numerator = (4 * alpha)**l
    denom = double_factorial(2*a - 1) * double_factorial(2*b - 1) * double_factorial(2*c - 1)
    return prefactor * np.sqrt(numerator / denom)

# ---------------------------------------------
# Primitive Gaussian with given (a, b, c)
# ---------------------------------------------
def gaussian_primitive(alpha, x, y, z, a, b, c, center):
    x0, y0, z0 = center
    xs, ys, zs = x-x0, y-y0, z-z0
    r2 = xs**2 + ys**2 + zs**2
    norm = norm_cartesian_gaussian(alpha, a, b, c)
    return norm * (xs**a) * (ys**b) * (zs**c) * np.exp(-alpha * r2)

# ---------------------------------------------
# Build AOs
# ---------------------------------------------
def AO_norm(primitives, coeffs, dV):
    norm = 0.0
    n = len(coeffs)
    for i in range(n):
        for j in range(n):
            overlap = np.sum(primitives[i] * primitives[j]) * dV
            norm += coeffs[i] * coeffs[j] * overlap
    return 1.0 / np.sqrt(norm)

def AO(alphas, coeffs, a, b, c, center, x, y, z, dV):
    primitives = [gaussian_primitive(alpha, x, y, z, a, b, c, center) for alpha in alphas]
    ao = sum(c * p for c, p in zip(coeffs, primitives))
    ao_norm_const = AO_norm(primitives, coeffs, dV)
    return ao_norm_const * ao

# ---------------------------------------------
# Build DO
# ---------------------------------------------
bl_A = 1.16  # Bond length in angstroms
bl_au = bl_A / 52.9e-2  # bond length in au
a = bl_au / 2
R_C = np.array([0, 0, a])
R_N = np.array([0, 0, -a])

def recenter_DO(DO, x, y, z, dV):
    density = np.abs(DO)**2
    x_c = np.sum(density * x) * dV
    y_c = np.sum(density * y) * dV
    z_c = np.sum(density * z) * dV
    return np.array([x_c, y_c, z_c])

# --- C S shells ---
c_alpha_1S = np.array([8236.0, 1235.0, 280.8, 79.27, 25.59, 8.997, 3.319, 0.3643])
c_coeffs_1S = np.array([5.31e-4, 4.108e-3, 2.1087e-2, 8.1853e-2, 0.234817, 0.434401, 0.346129, -8.983e-3])

c_alpha_2S = np.array([8236.0, 1235.0, 280.8, 79.27, 25.59, 8.997, 3.319, 0.3643])
c_coeffs_2S = np.array([-1.13e-4, -8.78e-4, -4.54e-3, -1.8133e-2, -5.576e-2, -0.126895, -0.170352, 0.598684])

c_alpha_3S = np.array([0.9059])
c_coeffs_3S = np.array([1.0])

c_alpha_4S = np.array([0.1285])
c_coeffs_4S = np.array([1.0])

c_alpha_5S = np.array([0.04402])
c_coeffs_5S = np.array([1.0])

# --- C P shells ---
c_alpha_1P = np.array([18.71, 4.133, 1.2])
c_coeffs_1P = np.array([0.014031, 0.086866, 0.290216])

c_alpha_2P = np.array([0.3827])
c_coeffs_2P = np.array([1.0])

c_alpha_3P = np.array([0.1209])
c_coeffs_3P = np.array([1.0])

c_alpha_4P = np.array([0.03569])
c_coeffs_4P = np.array([1.0])

# --- C D shells ---
c_alpha_1D = np.array([1.097])
c_coeffs_1D = np.array([1.0])

c_alpha_2D = np.array([0.318])
c_coeffs_2D = np.array([1.0])

c_alpha_3D = np.array([0.1])
c_coeffs_3D = np.array([1.0])

# --- C F shells ---
c_alpha_1F = np.array([0.761])
c_coeffs_1F = np.array([1.0])

c_alpha_2F = np.array([0.268])
c_coeffs_2F = np.array([1.0])

# --- N S shells ---
n_alpha_1S = np.array([11420.0, 1712.0, 389.3, 110.0, 35.57, 12.54, 4.644, 0.5118])
n_coeffs_1S = np.array([5.23e-4, 4.045e-3, 2.0775e-2, 8.0727e-2, 0.233074, 0.433501, 0.347472, -8.508e-3])

n_alpha_2S = np.array([11420.0, 1712.0, 389.3, 110.0, 35.57, 12.54, 4.644, 0.5118])
n_coeffs_2S = np.array([-1.15e-4, -8.95e-4, -4.624e-3, -1.8528e-2, -5.7339e-2, -0.132076, -0.17251, 0.599944])

n_alpha_3S = np.array([1.293])
n_coeffs_3S = np.array([1.0])

n_alpha_4S = np.array([0.1787])
n_coeffs_4S = np.array([1.0])

n_alpha_5S = np.array([0.0576])
n_coeffs_5S = np.array([1.0])

# --- N P shells ---
n_alpha_1P = np.array([26.63, 5.948, 1.742])
n_coeffs_1P = np.array([0.01467, 0.091764, 0.298683])

n_alpha_2P = np.array([0.555])
n_coeffs_2P = np.array([1.0])

n_alpha_3P = np.array([0.1725])
n_coeffs_3P = np.array([1.0])

n_alpha_4P = np.array([0.0491])
n_coeffs_4P = np.array([1.0])

# --- N D shells ---
n_alpha_1D = np.array([1.654])
n_coeffs_1D = np.array([1.0])

n_alpha_2D = np.array([0.469])
n_coeffs_2D = np.array([1.0])

n_alpha_3D = np.array([0.151])
n_coeffs_3D = np.array([1.0])

# --- N F shells ---
n_alpha_1F = np.array([1.093])
n_coeffs_1F = np.array([1.0])

n_alpha_2F = np.array([0.364])
n_coeffs_2F = np.array([1.0])

DO_coeffs_L = np.array([
     3.16457395e-04,
    -1.89930358e-01,
    -3.50233243e-02,
    -3.58713314e-01,
    -2.49990689e-01,
    -3.03663240e-15,
     6.41779414e-15,
    -2.42500851e-01,
    -4.75820011e-15,
     4.41906834e-14,
    -2.79193372e-01,
     8.81137522e-16,
     8.64041334e-14,
    -1.47366462e-01,
    -1.05372968e-15,
    -2.14744508e-14,
    -2.78468517e-02,
    -6.98568243e-17,
     1.24730248e-15,
     1.85075949e-02,
    -4.28511758e-17,
     1.54730177e-12,
    -2.34157912e-16,
    -5.63944497e-14,
     5.65877551e-03,
     9.59130747e-16,
    -3.99306125e-12,
     2.56604399e-16,
     2.93046446e-14,
    -1.34127809e-02,
    -1.85117139e-15,
     4.32062498e-11,
     4.88826008e-18,
    -3.01323087e-17,
     6.25882949e-15,
    -4.30515578e-03,
     1.49122565e-16,
     2.83627789e-14,
    -1.26459852e-17,
     9.05400608e-17,
     1.26815467e-16,
     1.12414448e-14,
     7.69190266e-03,
    -1.02087070e-16,
     1.30995702e-12,
     1.24571426e-16,
     3.78215640e-04,
    -9.52229509e-04,
     5.37506886e-03,
    -6.36037424e-02,
    -1.11538630e-02,
    -4.00728310e-15,
     6.20403387e-15,
     1.90263932e-01,
    -4.07635585e-15,
     6.43897039e-15,
     2.19452805e-01,
    -8.49060431e-15,
    -1.20313677e-13,
     1.51555287e-01,
    -1.04765063e-15,
     2.04918419e-15,
     5.50777242e-02,
    -7.70158490e-17,
     8.47518635e-16,
     8.88876540e-03,
    -2.19315244e-16,
     9.03595676e-14,
     6.05136703e-16,
    -2.51468644e-14,
     7.68754620e-03,
    -6.47600463e-16,
     1.35543522e-12,
    -1.32311267e-15,
    -4.79881527e-14,
    -4.87346023e-03,
    -2.55342304e-15,
    -4.03394201e-12,
     2.18537066e-16,
    -5.61713048e-17,
    -1.56203265e-15,
     1.76705276e-03,
    -1.67033786e-16,
     2.76310503e-13,
     4.38463794e-17,
    -1.56201810e-16,
     2.83876725e-16,
    -1.45705647e-14,
     1.44059310e-03,
    -4.71456323e-16,
    -1.72661546e-12,
    -9.11873038e-17
])

norm_L = 0.9492

DO_coeffs_R = np.array([
    -7.44795928e-06,
    -1.89986581e-01,
    -3.52877156e-02,
    -3.58001633e-01,
    -2.40966948e-01,
    -2.86794246e-15,
    5.42610074e-15,
    -2.40969120e-01,
    -5.71568106e-15,
    2.10538992e-14,
    -2.80151343e-01,
    -1.52887894e-15,
    1.24146531e-14,
    -1.49583776e-01,
    -1.33355472e-15,
    -2.66433710e-14,
    -2.81279911e-02,
    -7.00415150e-17,
    9.48614459e-16,
    1.90613388e-02,
    6.74965305e-17,
    1.51450585e-12,
    -3.71296890e-16,
    -3.08967367e-14,
    7.07509010e-03,
    1.46565686e-15,
    -3.76905492e-12,
    2.88010857e-16,
    5.72229519e-14,
    -1.27558660e-02,
    -6.89664831e-16,
    4.25353085e-11,
    -1.26981412e-17,
    -3.96918101e-18,
    4.77450703e-15,
    -4.80944406e-03,
    -3.74531902e-18,
    3.77361194e-14,
    -2.03180916e-17,
    7.85595898e-17,
    2.41679452e-16,
    -2.57627668e-15,
    7.42399997e-03,
    -1.43777137e-16,
    1.30359998e-12,
    1.24398636e-16,
    5.04272426e-05,
    -3.18006425e-03,
    4.00801753e-03,
    -6.70167109e-02,
    -1.38558028e-02,
    -4.16854704e-15,
    2.30678269e-15,
    1.90949090e-01,
    -4.54672669e-15,
    6.27792961e-17,
    2.22751816e-01,
    -5.22868482e-15,
    -2.02409163e-14,
    1.52259773e-01,
    -3.87792537e-16,
    1.66232483e-14,
    5.18366878e-02,
    -7.00651963e-17,
    2.51514408e-17,
    9.15228869e-03,
    -2.44194802e-16,
    9.16220356e-14,
    6.97973958e-16,
    -1.15258857e-14,
    8.02230448e-03,
    -4.79019912e-16,
    1.38830659e-12,
    -1.24848608e-15,
    -1.49016592e-14,
    -5.38455344e-03,
    -1.05611954e-15,
    -4.11764600e-12,
    2.29500182e-16,
    -6.39377552e-17,
    -9.92413523e-16,
    1.87635685e-03,
    -9.78234408e-17,
    2.91068487e-13,
    4.34705307e-17,
    -1.62594082e-16,
    4.07010786e-16,
    -1.11453300e-14,
    1.26176589e-03,
    -9.00676731e-17,
    -1.74207006e-12,
    -8.61067160e-17
])
norm_R = 0.9365

def build_DO(DO_coeffs, x, y, z, dV, recenter=True, print_geom=False):
    global R_C, R_N

    basis_info = [
    # ---- Carbon S orbitals ----
        (c_alpha_1S, c_coeffs_1S, 0, 0, 0, R_C, 0),
        (c_alpha_2S, c_coeffs_2S, 0, 0, 0, R_C, 1),
        (c_alpha_3S, c_coeffs_3S, 0, 0, 0, R_C, 2),
        (c_alpha_4S, c_coeffs_4S, 0, 0, 0, R_C, 3),
        (c_alpha_5S, c_coeffs_5S, 0, 0, 0, R_C, 4),

    # ---- Carbon P orbitals ----
        (c_alpha_1P, c_coeffs_1P, 1, 0, 0, R_C, 5),
        (c_alpha_1P, c_coeffs_1P, 0, 1, 0, R_C, 6),
        (c_alpha_1P, c_coeffs_1P, 0, 0, 1, R_C, 7),

        (c_alpha_2P, c_coeffs_2P, 1, 0, 0, R_C, 8),
        (c_alpha_2P, c_coeffs_2P, 0, 1, 0, R_C, 9),
        (c_alpha_2P, c_coeffs_2P, 0, 0, 1, R_C, 10),

        (c_alpha_3P, c_coeffs_3P, 1, 0, 0, R_C, 11),
        (c_alpha_3P, c_coeffs_3P, 0, 1, 0, R_C, 12),
        (c_alpha_3P, c_coeffs_3P, 0, 0, 1, R_C, 13),

        (c_alpha_4P, c_coeffs_4P, 1, 0, 0, R_C, 14),
        (c_alpha_4P, c_coeffs_4P, 0, 1, 0, R_C, 15),
        (c_alpha_4P, c_coeffs_4P, 0, 0, 1, R_C, 16),

    # ---- Carbon D orbitals ----
        (c_alpha_1D, c_coeffs_1D, 1, 1, 0, R_C, 17),
        (c_alpha_1D, c_coeffs_1D, 0, 1, 1, R_C, 18),
    # d_z^2 (19) done manually
        (c_alpha_1D, c_coeffs_1D, 1, 0, 1, R_C, 20),
    # d_x2-y2 (21) done manually

        (c_alpha_2D, c_coeffs_2D, 1, 1, 0, R_C, 22),
        (c_alpha_2D, c_coeffs_2D, 0, 1, 1, R_C, 23),
    # d_z^2 (24) done manually
        (c_alpha_2D, c_coeffs_2D, 1, 0, 1, R_C, 25),
    # d_x2-y2 (26) done manually

        (c_alpha_3D, c_coeffs_3D, 1, 1, 0, R_C, 27),
        (c_alpha_3D, c_coeffs_3D, 0, 1, 1, R_C, 28),
    # d_z^2 (29) done manually
        (c_alpha_3D, c_coeffs_3D, 1, 0, 1, R_C, 30),
    # d_x2-y2 (31) done manually

    # ---- Carbon F orbitals ----
    # f_y3x2 (32) done manually
        (c_alpha_1F, c_coeffs_1F, 1, 1, 1, R_C, 33),
    # f_yz2 (34) done manually
    # f_z3 (35) done manually
    # f_xz2 (36) done manually
    # f_x2y2z (37) done manually
    # f_x3y2 (38) done manually

    # f_y3x2 (39) done manually
        (c_alpha_2F, c_coeffs_2F, 1, 1, 1, R_C, 40),
    # f_yz2 (41) done manually
    # f_z3 (42) done manually
    # f_xz2 (43) done manually
    # f_x2y2z (44) done manually
    # f_x3y2 (45) done manually

    # ---- Nitrogen S orbitals ----
        (n_alpha_1S, n_coeffs_1S, 0, 0, 0, R_N, 46),
        (n_alpha_2S, n_coeffs_2S, 0, 0, 0, R_N, 47),
        (n_alpha_3S, n_coeffs_3S, 0, 0, 0, R_N, 48),
        (n_alpha_4S, n_coeffs_4S, 0, 0, 0, R_N, 49),
        (n_alpha_5S, n_coeffs_5S, 0, 0, 0, R_N, 50),

    # ---- Nitrogen P orbitals ----
        (n_alpha_1P, n_coeffs_1P, 1, 0, 0, R_N, 51),
        (n_alpha_1P, n_coeffs_1P, 0, 1, 0, R_N, 52),
        (n_alpha_1P, n_coeffs_1P, 0, 0, 1, R_N, 53),

        (n_alpha_2P, n_coeffs_2P, 1, 0, 0, R_N, 54),
        (n_alpha_2P, n_coeffs_2P, 0, 1, 0, R_N, 55),
        (n_alpha_2P, n_coeffs_2P, 0, 0, 1, R_N, 56),

        (n_alpha_3P, n_coeffs_3P, 1, 0, 0, R_N, 57),
        (n_alpha_3P, n_coeffs_3P, 0, 1, 0, R_N, 58),
        (n_alpha_3P, n_coeffs_3P, 0, 0, 1, R_N, 59),

        (n_alpha_4P, n_coeffs_4P, 1, 0, 0, R_N, 60),
        (n_alpha_4P, n_coeffs_4P, 0, 1, 0, R_N, 61),
        (n_alpha_4P, n_coeffs_4P, 0, 0, 1, R_N, 62),

    # ---- Nitrogen D orbitals ----
        (n_alpha_1D, n_coeffs_1D, 1, 1, 0, R_N, 63),
        (n_alpha_1D, n_coeffs_1D, 0, 1, 1, R_N, 64),
    # d_z^2 (65) done manually
        (n_alpha_1D, n_coeffs_1D, 1, 0, 1, R_N, 66),
    # d_x2-y2 (67) done manually

        (n_alpha_2D, n_coeffs_2D, 1, 1, 0, R_N, 68),
        (n_alpha_2D, n_coeffs_2D, 0, 1, 1, R_N, 69),
    # d_z^2 (70) done manually
        (n_alpha_2D, n_coeffs_2D, 1, 0, 1, R_N, 71),
    # d_x2-y2 (72) done manually

        (n_alpha_3D, n_coeffs_3D, 1, 1, 0, R_N, 73),
        (n_alpha_3D, n_coeffs_3D, 0, 1, 1, R_N, 74),
    # d_z^2 (75) done manually
        (n_alpha_3D, n_coeffs_3D, 1, 0, 1, R_N, 76),
    # d_x2-y2 (77) done manually

    # ---- Nitrogen F orbitals ----
    # f_y3x2 (78) done manually
        (n_alpha_1F, n_coeffs_1F, 1, 1, 1, R_C, 79),
    # f_yz2 (80) done manually
    # f_z3 (81) done manually
    # f_xz2 (82) done manually
    # f_x2y2z (83) done manually
    # f_x3y2 (84) done manually

    # f_y3x2 (85) done manually
        (n_alpha_2F, n_coeffs_2F, 1, 1, 1, R_C, 86),
    # f_yz2 (87) done manually
    # f_z3 (88) done manually
    # f_xz2 (89) done manually
    # f_x2y2z (90) done manually
    # f_x3y2 (91) done manually
]
 
    DO = np.zeros_like(x, dtype=np.float64)
    threshold = 1e-5
    for alphas, coeffs, a, b, c, center, i in basis_info:
        coeff = DO_coeffs[i]
        if abs(coeff) >= threshold:
            ao = AO(alphas, coeffs, a, b, c, center, x, y, z, dV)
            DO += coeff * ao

    def norm(orb):
        return 1 / np.sqrt(integrate_3d(abs(orb)**2, dV))
    
    # Handle weird orbitals manually
    if abs(DO_coeffs[19]) >= threshold:
        ao_zz = AO(c_alpha_1D, c_coeffs_1D, 0, 0, 2, R_C, x, y, z, dV)
        ao_xx = AO(c_alpha_1D, c_coeffs_1D, 2, 0, 0, R_C, x, y, z, dV)
        ao_yy = AO(c_alpha_1D, c_coeffs_1D, 0, 2, 0, R_C, x, y, z, dV)
        orb = 0.5 * (2 * ao_zz - ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[19] * normed_AO

    if abs(DO_coeffs[21]) >= threshold:
        ao_xx = AO(c_alpha_1D, c_coeffs_1D, 2, 0, 0, R_C, x, y, z, dV)
        ao_yy = AO(c_alpha_1D, c_coeffs_1D, 0, 2, 0, R_C, x, y, z, dV)
        orb = (np.sqrt(3)/2) * (ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb   
        DO += DO_coeffs[21] * normed_AO

    if abs(DO_coeffs[24]) >= threshold:
        ao_zz = AO(c_alpha_2D, c_coeffs_2D, 0, 0, 2, R_C, x, y, z, dV)
        ao_xx = AO(c_alpha_2D, c_coeffs_2D, 2, 0, 0, R_C, x, y, z, dV)
        ao_yy = AO(c_alpha_2D, c_coeffs_2D, 0, 2, 0, R_C, x, y, z, dV)
        orb = 0.5 * (2 * ao_zz - ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[24] * normed_AO

    if abs(DO_coeffs[26]) >= threshold:
        ao_xx = AO(c_alpha_2D, c_coeffs_2D, 2, 0, 0, R_C, x, y, z, dV)
        ao_yy = AO(c_alpha_2D, c_coeffs_2D, 0, 2, 0, R_C, x, y, z, dV)
        orb = (np.sqrt(3)/2) * (ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb   
        DO += DO_coeffs[26] * normed_AO

    if abs(DO_coeffs[29]) >= threshold:
        ao_zz = AO(c_alpha_3D, c_coeffs_3D, 0, 0, 2, R_C, x, y, z, dV)
        ao_xx = AO(c_alpha_3D, c_coeffs_3D, 2, 0, 0, R_C, x, y, z, dV)
        ao_yy = AO(c_alpha_3D, c_coeffs_3D, 0, 2, 0, R_C, x, y, z, dV)
        orb = 0.5 * (2 * ao_zz - ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[29] * normed_AO

    if abs(DO_coeffs[31]) >= threshold:
        ao_xx = AO(c_alpha_3D, c_coeffs_3D, 2, 0, 0, R_C, x, y, z, dV)
        ao_yy = AO(c_alpha_3D, c_coeffs_3D, 0, 2, 0, R_C, x, y, z, dV)
        orb = (np.sqrt(3)/2) * (ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb   
        DO += DO_coeffs[31] * normed_AO

    if abs(DO_coeffs[32]) >= threshold:
        ao_yx2 = AO(c_alpha_1F, c_coeffs_1F, 2, 1, 0, R_C, x, y, z, dV)
        ao_y3  = AO(c_alpha_1F, c_coeffs_1F, 0, 3, 0, R_C, x, y, z, dV)
        orb = np.sqrt(5/8) * (3 * ao_yx2 - ao_y3)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[32] * normed_AO

    if abs(DO_coeffs[34]) >= threshold:
        ao_yz2 = AO(c_alpha_1F, c_coeffs_1F, 0, 1, 2, R_C, x, y, z, dV)
        ao_y3  = AO(c_alpha_1F, c_coeffs_1F, 0, 3, 0, R_C, x, y, z, dV)
        ao_yx2 = AO(c_alpha_1F, c_coeffs_1F, 2, 1, 0, R_C, x, y, z, dV)
        orb = np.sqrt(3/8) * (4 * ao_yz2 - ao_y3 - ao_yx2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[34] * normed_AO

    if abs(DO_coeffs[35]) >= threshold:
        ao_z3   = AO(c_alpha_1F, c_coeffs_1F, 0, 0, 3, R_C, x, y, z, dV)
        ao_x2z  = AO(c_alpha_1F, c_coeffs_1F, 2, 0, 1, R_C, x, y, z, dV)
        ao_y2z  = AO(c_alpha_1F, c_coeffs_1F, 0, 2, 1, R_C, x, y, z, dV)
        orb = 0.5 * (2 * ao_z3 - 3 * ao_x2z - 3 * ao_y2z)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[35] * normed_AO

    if abs(DO_coeffs[36]) >= threshold:
        ao_xz2 = AO(c_alpha_1F, c_coeffs_1F, 1, 0, 2, R_C, x, y, z, dV)
        ao_x3   = AO(c_alpha_1F, c_coeffs_1F, 3, 0, 0, R_C, x, y, z, dV)
        ao_xy2  = AO(c_alpha_1F, c_coeffs_1F, 1, 2, 0, R_C, x, y, z, dV)
        orb = np.sqrt(3/8) * (4 * ao_xz2 - ao_x3 - ao_xy2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[36] * normed_AO

    if abs(DO_coeffs[37]) >= threshold:
        ao_x2z = AO(c_alpha_1F, c_coeffs_1F, 2, 0, 1, R_C, x, y, z, dV)
        ao_y2z = AO(c_alpha_1F, c_coeffs_1F, 0, 2, 1, R_C, x, y, z, dV)
        orb = (np.sqrt(15) / 2) * (ao_x2z - ao_y2z)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[37] * normed_AO

    if abs(DO_coeffs[38]) >= threshold:
        ao_x3 = AO(c_alpha_1F, c_coeffs_1F, 3, 0, 0, R_C, x, y, z, dV)
        ao_xy2 = AO(c_alpha_1F, c_coeffs_1F, 1, 2, 0, R_C, x, y, z, dV)
        orb = np.sqrt(5/8) * (ao_x3 - 3 * ao_xy2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[38] * normed_AO

    if abs(DO_coeffs[39]) >= threshold:
        ao_yx2 = AO(c_alpha_2F, c_coeffs_2F, 2, 1, 0, R_C, x, y, z, dV)
        ao_y3  = AO(c_alpha_2F, c_coeffs_2F, 0, 3, 0, R_C, x, y, z, dV)
        orb = np.sqrt(5/8) * (3 * ao_yx2 - ao_y3)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[39] * normed_AO

    if abs(DO_coeffs[41]) >= threshold:
        ao_yz2 = AO(c_alpha_2F, c_coeffs_2F, 0, 1, 2, R_C, x, y, z, dV)
        ao_y3  = AO(c_alpha_2F, c_coeffs_2F, 0, 3, 0, R_C, x, y, z, dV)
        ao_yx2 = AO(c_alpha_2F, c_coeffs_2F, 2, 1, 0, R_C, x, y, z, dV)
        orb = np.sqrt(3/8) * (4 * ao_yz2 - ao_y3 - ao_yx2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[41] * normed_AO

    if abs(DO_coeffs[42]) >= threshold:
        ao_z3   = AO(c_alpha_2F, c_coeffs_2F, 0, 0, 3, R_C, x, y, z, dV)
        ao_x2z  = AO(c_alpha_2F, c_coeffs_2F, 2, 0, 1, R_C, x, y, z, dV)
        ao_y2z  = AO(c_alpha_2F, c_coeffs_2F, 0, 2, 1, R_C, x, y, z, dV)
        orb = 0.5 * (2 * ao_z3 - 3 * ao_x2z - 3 * ao_y2z)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[42] * normed_AO

    if abs(DO_coeffs[43]) >= threshold:
        ao_xz2 = AO(c_alpha_2F, c_coeffs_2F, 1, 0, 2, R_C, x, y, z, dV)
        ao_x3   = AO(c_alpha_2F, c_coeffs_2F, 3, 0, 0, R_C, x, y, z, dV)
        ao_xy2  = AO(c_alpha_2F, c_coeffs_2F, 1, 2, 0, R_C, x, y, z, dV)
        orb = np.sqrt(3/8) * (4 * ao_xz2 - ao_x3 - ao_xy2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[43] * normed_AO

    if abs(DO_coeffs[44]) >= threshold:
        ao_x2z = AO(c_alpha_2F, c_coeffs_2F, 2, 0, 1, R_C, x, y, z, dV)
        ao_y2z = AO(c_alpha_2F, c_coeffs_2F, 0, 2, 1, R_C, x, y, z, dV)
        orb = (np.sqrt(15) / 2) * (ao_x2z - ao_y2z)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[44] * normed_AO

    if abs(DO_coeffs[45]) >= threshold:
        ao_x3 = AO(c_alpha_2F, c_coeffs_2F, 3, 0, 0, R_C, x, y, z, dV)
        ao_xy2 = AO(c_alpha_2F, c_coeffs_2F, 1, 2, 0, R_C, x, y, z, dV)
        orb = np.sqrt(5/8) * (ao_x3 - 3 * ao_xy2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[45] * normed_AO

    if abs(DO_coeffs[65]) >= threshold:
        ao_zz = AO(n_alpha_1D, n_coeffs_1D, 0, 0, 2, R_N, x, y, z, dV)
        ao_xx = AO(n_alpha_1D, n_coeffs_1D, 2, 0, 0, R_N, x, y, z, dV)
        ao_yy = AO(n_alpha_1D, n_coeffs_1D, 0, 2, 0, R_N, x, y, z, dV)
        orb = 0.5 * (2 * ao_zz - ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[65] * normed_AO

    if abs(DO_coeffs[67]) >= threshold:
        ao_xx = AO(n_alpha_1D, n_coeffs_1D, 2, 0, 0, R_N, x, y, z, dV)
        ao_yy = AO(n_alpha_1D, n_coeffs_1D, 0, 2, 0, R_N, x, y, z, dV)
        orb = (np.sqrt(3)/2) * (ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb   
        DO += DO_coeffs[67] * normed_AO

    if abs(DO_coeffs[70]) >= threshold:
        ao_zz = AO(n_alpha_2D, n_coeffs_2D, 0, 0, 2, R_N, x, y, z, dV)
        ao_xx = AO(n_alpha_2D, n_coeffs_2D, 2, 0, 0, R_N, x, y, z, dV)
        ao_yy = AO(n_alpha_2D, n_coeffs_2D, 0, 2, 0, R_N, x, y, z, dV)
        orb = 0.5 * (2 * ao_zz - ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[70] * normed_AO

    if abs(DO_coeffs[72]) >= threshold:
        ao_xx = AO(n_alpha_2D, n_coeffs_2D, 2, 0, 0, R_N, x, y, z, dV)
        ao_yy = AO(n_alpha_2D, n_coeffs_2D, 0, 2, 0, R_N, x, y, z, dV)
        orb = (np.sqrt(3)/2) * (ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb   
        DO += DO_coeffs[72] * normed_AO

    if abs(DO_coeffs[75]) >= threshold:
        ao_zz = AO(n_alpha_3D, n_coeffs_3D, 0, 0, 2, R_N, x, y, z, dV)
        ao_xx = AO(n_alpha_3D, n_coeffs_3D, 2, 0, 0, R_N, x, y, z, dV)
        ao_yy = AO(n_alpha_3D, n_coeffs_3D, 0, 2, 0, R_N, x, y, z, dV)
        orb = 0.5 * (2 * ao_zz - ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[75] * normed_AO

    if abs(DO_coeffs[77]) >= threshold:
        ao_xx = AO(n_alpha_3D, n_coeffs_3D, 2, 0, 0, R_N, x, y, z, dV)
        ao_yy = AO(n_alpha_3D, n_coeffs_3D, 0, 2, 0, R_N, x, y, z, dV)
        orb = (np.sqrt(3)/2) * (ao_xx - ao_yy)
        N = norm(orb)
        normed_AO = N * orb   
        DO += DO_coeffs[77] * normed_AO

    if abs(DO_coeffs[78]) >= threshold:
        ao_yx2 = AO(n_alpha_1F, n_coeffs_1F, 2, 1, 0, R_N, x, y, z, dV)
        ao_y3  = AO(n_alpha_1F, n_coeffs_1F, 0, 3, 0, R_N, x, y, z, dV)
        orb = np.sqrt(5/8) * (3 * ao_yx2 - ao_y3)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[78] * normed_AO

    if abs(DO_coeffs[80]) >= threshold:
        ao_yz2 = AO(n_alpha_1F, n_coeffs_1F, 0, 1, 2, R_N, x, y, z, dV)
        ao_y3  = AO(n_alpha_1F, n_coeffs_1F, 0, 3, 0, R_N, x, y, z, dV)
        ao_yx2 = AO(n_alpha_1F, n_coeffs_1F, 2, 1, 0, R_N, x, y, z, dV)
        orb = np.sqrt(3/8) * (4 * ao_yz2 - ao_y3 - ao_yx2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[80] * normed_AO

    if abs(DO_coeffs[81]) >= threshold:
        ao_z3   = AO(n_alpha_1F, n_coeffs_1F, 0, 0, 3, R_N, x, y, z, dV)
        ao_x2z  = AO(n_alpha_1F, n_coeffs_1F, 2, 0, 1, R_N, x, y, z, dV)
        ao_y2z  = AO(n_alpha_1F, n_coeffs_1F, 0, 2, 1, R_N, x, y, z, dV)
        orb = 0.5 * (2 * ao_z3 - 3 * ao_x2z - 3 * ao_y2z)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[81] * normed_AO

    if abs(DO_coeffs[82]) >= threshold:
        ao_xz2 = AO(n_alpha_1F, n_coeffs_1F, 1, 0, 2, R_N, x, y, z, dV)
        ao_x3   = AO(n_alpha_1F, n_coeffs_1F, 3, 0, 0, R_N, x, y, z, dV)
        ao_xy2  = AO(n_alpha_1F, n_coeffs_1F, 1, 2, 0, R_N, x, y, z, dV)
        orb = np.sqrt(3/8) * (4 * ao_xz2 - ao_x3 - ao_xy2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[82] * normed_AO

    if abs(DO_coeffs[83]) >= threshold:
        ao_x2z = AO(n_alpha_1F, n_coeffs_1F, 2, 0, 1, R_N, x, y, z, dV)
        ao_y2z = AO(n_alpha_1F, n_coeffs_1F, 0, 2, 1, R_N, x, y, z, dV)
        orb = (np.sqrt(15) / 2) * (ao_x2z - ao_y2z)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[83] * normed_AO

    if abs(DO_coeffs[84]) >= threshold:
        ao_x3 = AO(n_alpha_1F, n_coeffs_1F, 3, 0, 0, R_N, x, y, z, dV)
        ao_xy2 = AO(n_alpha_1F, n_coeffs_1F, 1, 2, 0, R_N, x, y, z, dV)
        orb = np.sqrt(5/8) * (ao_x3 - 3 * ao_xy2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[84] * normed_AO

    if abs(DO_coeffs[85]) >= threshold:
        ao_yx2 = AO(n_alpha_2F, n_coeffs_2F, 2, 1, 0, R_N, x, y, z, dV)
        ao_y3  = AO(n_alpha_2F, n_coeffs_2F, 0, 3, 0, R_N, x, y, z, dV)
        orb = np.sqrt(5/8) * (3 * ao_yx2 - ao_y3)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[85] * normed_AO

    if abs(DO_coeffs[87]) >= threshold:
        ao_yz2 = AO(n_alpha_2F, n_coeffs_2F, 0, 1, 2, R_N, x, y, z, dV)
        ao_y3  = AO(n_alpha_2F, n_coeffs_2F, 0, 3, 0, R_N, x, y, z, dV)
        ao_yx2 = AO(n_alpha_2F, n_coeffs_2F, 2, 1, 0, R_N, x, y, z, dV)
        orb = np.sqrt(3/8) * (4 * ao_yz2 - ao_y3 - ao_yx2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[87] * normed_AO

    if abs(DO_coeffs[88]) >= threshold:
        ao_z3   = AO(n_alpha_2F, n_coeffs_2F, 0, 0, 3, R_N, x, y, z, dV)
        ao_x2z  = AO(n_alpha_2F, n_coeffs_2F, 2, 0, 1, R_N, x, y, z, dV)
        ao_y2z  = AO(n_alpha_2F, n_coeffs_2F, 0, 2, 1, R_N, x, y, z, dV)
        orb = 0.5 * (2 * ao_z3 - 3 * ao_x2z - 3 * ao_y2z)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[88] * normed_AO

    if abs(DO_coeffs[89]) >= threshold:
        ao_xz2 = AO(n_alpha_2F, n_coeffs_2F, 1, 0, 2, R_N, x, y, z, dV)
        ao_x3   = AO(n_alpha_2F, n_coeffs_2F, 3, 0, 0, R_N, x, y, z, dV)
        ao_xy2  = AO(n_alpha_2F, n_coeffs_2F, 1, 2, 0, R_N, x, y, z, dV)
        orb = np.sqrt(3/8) * (4 * ao_xz2 - ao_x3 - ao_xy2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[89] * normed_AO

    if abs(DO_coeffs[90]) >= threshold:
        ao_x2z = AO(n_alpha_2F, n_coeffs_2F, 2, 0, 1, R_N, x, y, z, dV)
        ao_y2z = AO(n_alpha_2F, n_coeffs_2F, 0, 2, 1, R_N, x, y, z, dV)
        orb = (np.sqrt(15) / 2) * (ao_x2z - ao_y2z)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[90] * normed_AO

    if abs(DO_coeffs[91]) >= threshold:
        ao_x3 = AO(n_alpha_2F, n_coeffs_2F, 3, 0, 0, R_N, x, y, z, dV)
        ao_xy2 = AO(n_alpha_2F, n_coeffs_2F, 1, 2, 0, R_N, x, y, z, dV)
        orb = np.sqrt(5/8) * (ao_x3 - 3 * ao_xy2)
        N = norm(orb)
        normed_AO = N * orb
        DO += DO_coeffs[91] * normed_AO

    Norm = norm(DO)
    DO *= Norm

    if recenter:
        centroid = recenter_DO(DO, x, y, z, dV)
        
        # Shift grid
        x_new = x - centroid[0]
        y_new = y - centroid[1]
        z_new = z - centroid[2]

        # Shift molecular geometry
        R_C = R_C - centroid
        R_N = R_N - centroid

        if print_geom:
            bohr_to_angstrom = 0.529177
            R_C_ang = R_C * bohr_to_angstrom
            R_N_ang = R_N * bohr_to_angstrom

            print("\nShifting molecular geometry to the new center")
            print("New molecular geometry is (in Ångströms):")
            print(" atom         X             Y             Z")
            print("   1      {: .6f}      {: .6f}      {: .6f}".format(*R_C_ang))
            print("   2      {: .6f}      {: .6f}      {: .6f}".format(*R_N_ang))

        # Rebuild on shifted grid
        return build_DO(DO_coeffs, x_new, y_new, z_new, dV, recenter=False)

    return DO, R_C, R_N