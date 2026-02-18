# Dyson Orbital Photodetachment Calculator

A high-performance C++/Python suite for calculating photodetachment cross sections ($\sigma$) and photoelectron angular distributions ($\beta$) from molecular Dyson orbitals computed by Q-Chem.

## Features

*   **Unified Workflow**: A single JSON configuration file controls the entire pipeline: Dyson orbital extraction, grid generation, physics calculation, and visualization.
*   **Multiple Continuum Models**:
    *   **Plane Wave Expansion (PWE)**: Fast analytic approach for $D \approx 0$, or rigorous numeric averaging for general cases.
    *   **Point Dipole**: Exact solution for the electron-dipole interaction ($1/r^2$ potential). Supports both analytic and numeric averaging.
    *   **Physical Dipole**: Regularized finite dipole model (charges $\pm q$ separated by $2a$), enabling calculations for super-critical dipole strengths ($D > D_{crit} = 1.279$) without collapse.
*   **Numeric Averaging**: Explicit orientation averaging on a Lebedev grid is now supported for all models, providing higher accuracy for anisotropic systems.
*   **Vibrational Resolution**: Support for calculating relative cross-sections across vibrational channels using Franck-Condon factors.
*   **Visualization**: Built-in tools for plotting Dyson orbitals and cross-section results.
*   **Documentation**: See `job_guide.txt` for a comprehensive step-by-step tutorial and configuration reference.

## Directory Structure

*   `code/cxx/`: Core physics engine (C++17). Handles computationally intensive grid integration and partial wave expansions.
*   `code/python/`: Driver scripts and I/O handlers. `run_job.py` is the main entry point.
*   `beta_gen`: Compiled binary for main calculations.
*   `dyson_gen`: Compiled binary for Dyson generation and cross-section sweeps.

## Installation

### Prerequisites
*   **C++ Compiler**: Must support C++17 (e.g., `g++` 9+, `clang++`).
*   **Python 3**: With `numpy`, `scipy`, `matplotlib`, and `pandas`.
*   **OpenMP**: Recommended for parallelizing grid integrations.

### Building
The project uses a standard `Makefile`. To build the binaries:

```bash
make
```
This produces `dyson_gen` and `beta_gen` in the root directory.

## Usage

The primary workflow is driven by `code/python/run_job.py` using a JSON config file.

```bash
python3 code/python/run_job.py job.json
```

### Example Configuration (`job.json`)

```json
{
  "qchem_output": "data/CuO_pVTZ.out",
  "dyson": {
    "do_generation": true,
    "indices": [0],
    "grid_step": 0.3,
    "padding": 20.0,
    "output_bin": "dyson.bin"
  },
  "calculation": {
    "do_calculation": true,
    "type": "cross_section",
    "model": "physical_dipole",
    "dipole": 0.64,
    "dipole_length": 0.1,
    "ie": 1.778,
    "energies": [0.1, 0.5, 1.0, 1.5],
    "output_csv": "results.csv"
  }
}
```

For a comprehensive guide on all configuration options, see `job_guide.txt`.

## Method Details

### Physical Dipole Model
For strong dipoles, the point dipole model exhibits a "fall-to-center" singularity. This code implements a **Finite Dipole** model where the potential is non-singular:
$$ V(\mathbf{r}) = -D \left( \frac{1}{|\mathbf{r} - \mathbf{a}|} - \frac{1}{|\mathbf{r} + \mathbf{a}|} \right) \frac{1}{2a} $$
(where $D$ is the dipole moment and separation is $2a$).

The radial equation is solved using:
1.  **Spherical Bessel Expansion** (Complex Order $\nu$) for high-energy/general cases.
2.  **Gallup Power Series** for near-threshold $m=0$ precision.

This allows for smooth evolution of cross-section features across the critical dipole threshold.