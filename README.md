# Multi-DCT: Low-Complexity Multidimensional DCT Approximations

This repository is a Python 3 implementation of the research presented in:
**"Low-Complexity Multidimensional DCT Approximations for High-Order Tensor Data Decorrelation"** by Vítor de A. Coutinho, Renato J. Cintra, and Fábio M. Bayer (IEEE Transactions on Image Processing, 2017).

## Overview
Standard 3D Discrete Cosine Transforms (DCT) are computationally expensive due to their multiplicative complexity. This project implements several multiplierless 3D DCT approximations derived through high-order tensor theory. These approximations are suitable for energy-constrained systems and real-time applications like video coding and visual tracking.

## Key Features
* **Multiplierless Computation:** Implements approximations (e.g., MRDCT, LODCT, BAS series) requiring only additions and bit-shifts.
* **Tensor-Based Framework:** Uses i-mode product formalism to generalize 1D approximations into 3D and higher dimensions.
* **Visual Tracking:** Includes a proof-of-concept discriminative tracker using 3D DCT subspace representation.
* **Modified Quantization:** A specialized procedure to absorb approximation scaling factors into the quantization volume.

## Project Organization
```
multi-dct-approx/
├── data/                       # Video sequences (e.g., 'animal', 'foreman')
├── src/
│   ├── core/                   # 1D Approximations and Tensor logic
│   │   ├── __init__.py
│   │   ├── approximations.py   # MRDCT, LODCT, etc.
│   │   └── tensor_ops.py       # i-mode products
│   ├── tracking/               # Visual tracking implementation
│   │   ├── __init__.py
│   │   └── tracker.py
│   └── utils/                  # Video I/O and visualization
├── tests/                      # Numerical verification against exact DCT
├── DESIGN.md
├── README.md
└── requirements.txt
```

## Getting Started
1. **Environment:** Setup via `pip install -r requirements.txt`.
2. **Core Verification:** Run `pytest` (to be implemented) to ensure `i_mode_product` correctly mimics the paper's tensor algebra.
3. **Current Focus:** Implementing 1D integer matrices ($T_N$) for MRDCT and LODCT.

## Verification Suite
This project uses `pytest` to ensure the approximations maintain the mathematical properties of the original DCT.

### Energy Compaction Test
Verifies that the approximation concentrates signal energy in low-frequency coefficients. 
**Status:** Implemented for MRDCT.

To run:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 tests/test_compaction.py
```