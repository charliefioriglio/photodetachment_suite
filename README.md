## Photodetachment Suite Overview

1. **Dyson-orbital ingestion and preparation.**  The `DO_reader` module parses
   QChem/ezDyson outputs, while curated builders such as `CN_do.py` and the
   visualization helpers in `visualizer/` provide ready-to-use reference
   orbitals.  All coefficients and primitive Gaussians originate from the
   ezDyson implementation shipped in `ezSuite/ezDyson_2021/ezdyson_code`, and we
   gratefully acknowledge that project for making the reference data available.
2. **Grid and orientational infrastructure.**  `grid.CartesianGrid` defines
   uniform 3D volumes, `integration.py` supplies trapz/Simpson quadrature, and
   `averaging_angle_grids/` offers hard-coded, simple, and repulsion-based Euler
   grids for orientational averaging.
3. **Continuum wavefunctions.**  `continuum.py` exposes analytic plane waves,
   plane-wave expansions, point-dipole continua (via SciPy/Wigner 3j), and the
   full physical-dipole solver implemented in `physical_dipole/`, which handles
   Gallup-style prolate-spheroidal modes with caching and mode summation.
4. **Observables.**  `cross_sections.py` and
   `physical_dipole/cross_sections.py` compute absolute or relative cross
   sections per channel, while `beta_calculator.py` returns anisotropy
   parameters together with parallel/perpendicular partials.  Helper scripts in
   `examples/` (e.g., `cn_do_full_workflow.py`) demonstrate end-to-end use.
5. **Visualization and regression tests.**  `plots.py`, the `visualizer/` CLI,
   and `results/` templates make it easy to inspect spectra, and the `tests/`
   directory contains fast regression suites to guard continuum and averaging
   logic.

Together these components let you swap Dyson sources, choose the appropriate
continuum model, and evaluate spectra or β scans with a single, reproducible
workflow.
