# Charged Elastic Rod (DNA Supercoiling) – BFGS Programming Assignment

This repo is a **starter kit** for a Numerical Methods programming assignment on **quasi-Newton optimization (BFGS)**.

Students will minimize the energy of a **closed, charged elastic filament** in 3D, discretized as `N` points.
The energy combines:
- **Bending** (curvature penalty),
- **Stretching** (near-inextensibility),
- **Screened Coulomb repulsion** (Debye–Hückel).

The core energy + gradient kernel is implemented in **C++** for speed and exposed to Python via **ctypes**.
You will implement **BFGS** (and a line search) in Python and use it to find low-energy configurations.

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2) Build the C++ shared library
Linux/macOS (clang/gcc):
```bash
bash csrc/build.sh
```

This produces `csrc/librod.so` (Linux) or `csrc/librod.dylib` (macOS).

### 3) Run a gradient check
```bash
pytest -q
```

### 4) Run the optimizer (after you implement BFGS)
```bash
python scripts/run_opt.py --N 120 --steps 200
```

## What you need to implement
- `src/elastic_rod/bfgs.py`: BFGS update + line search (Wolfe or backtracking).
- `csrc/rod_energy.cpp`: The code described in the TODO: comment.
- (Optional) better initialization and plotting in `scripts/run_opt.py`.

## Repo layout
- `csrc/`: C++ energy + gradient, builds a shared library
- `src/elastic_rod/`: Python wrapper + optimization code
- `docs/background.tex`: short LaTeX background note (assignment handout)
- `tests/`: finite-difference gradient checks

## Parameters (default)
- `kb`: bending stiffness
- `ks`: stretching stiffness
- `l0`: rest segment length
- `q`: charge magnitude per node
- `kappa`: screening parameter (0 gives unscreened Coulomb)

See `src/elastic_rod/model.py`.

## License
MIT (for the starter code). Add your course policy as needed.


## Gradescope
Staff: see `gradescope/README_GRADESCOPE.md` for an autograder skeleton.
Students: see `docs/AUTOGRADING.md` and run `python scripts/autograde_local.py --mode accuracy`.
