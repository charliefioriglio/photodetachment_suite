# Dyson Orbital Photodetachment Calculator

A high-performance C++/Python suite for calculating photodetachment cross sections ($\sigma$) and photoelectron angular distributions ($\beta$) from molecular Dyson orbitals computed by Q-Chem.

## Features

*   **Unified Workflow**: A single JSON configuration file controls the entire pipeline: Dyson orbital extraction, grid generation, physics calculation, and visualization.
*   **Multiple Continuum Models**:
    *   **Plane Wave Expansion (PWE)**: Fast analytic approach for $D \approx 0$, or rigorous numeric averaging for general cases.
    *   **Point Dipole**: Exact solution for the electron-dipole interaction ($1/r^2$ potential). Supports both analytic and numeric averaging.
    *   **Physical Dipole**: Regularized finite dipole model (charges $\pm q$ separated by $2a$), enabling calculations for super-critical dipole strengths ($D > D_{crit}$) without collapse.
*   **Averaging Options**: Program supports analytic averaging using Wigner-D matricies where applicable. For other cases, program supports numeric averaging witha  specifiable number of Euler Angles.
*   **Relative Vibrational Channel Cross Sections**: Support for calculating relative cross-sections across vibrational channels using Franck-Condon factors.
*   **Visualization**: Built-in tools for plotting Dyson orbitals and cross-section results.
*   **Beta Parameter Plotting**: Automated generation of Beta vs eKE plots with configurable output options.
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
*   **OpenMP**: Recommended for parallelizing grid integrations. Code will run without it.

### Building
The project uses a standard `Makefile`. To build the binaries:

```bash
make
```
This produces `dyson_gen` and `beta_gen` in the root directory.

## Usage

The primary workflow is driven by `code/python/run_job.py` using a JSON configuration file.

### Quick Start

1. Prepare a Q-Chem output file with Dyson orbital data.
2. Create a job configuration JSON (for example, `job.json`).
3. Run:

```bash
python3 code/python/run_job.py job.json
```

4. Review generated CSV and optional plot outputs.

For full configuration details, option tables, and complete examples, see [job_guide.txt](job_guide.txt).

## Documentation

- Full workflow and configuration reference: [job_guide.txt](job_guide.txt)
- Core implementation: `code/cxx/`
- Python driver and helpers: `code/python/`

## License and Attribution

This project is distributed under the GNU General Public License (GPL).
See `LICENSE` for terms.

### Provenance

This codebase includes components derived from ezDyson.
See `NOTICE` for file-level provenance, credit, and citation details.

### Citation

If you use this software in research, please cite this project and the ezDyson references listed in `NOTICE`.

For work completed before manuscript submission/publication, cite the software repository and reference the manuscript as "in preparation" where appropriate. After publication, add the full paper citation (authors, journal, year, DOI) in this README and in `NOTICE`.

### Collaboration

This work is being developed in collaboration with the Mabbs Group, Department of Chemistry, Washington University in St. Louis.
