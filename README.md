# Multi-DCT: Low-Complexity Multidimensional DCT Approximations

This repository is a Python 3 implementation of the research presented in:
**"Low-Complexity Multidimensional DCT Approximations for High-Order Tensor Data Decorrelation"** by Vítor de A. Coutinho, Renato J. Cintra, and Fábio M. Bayer (IEEE Transactions on Image Processing, 2017).

## Overview
Standard 3D Discrete Cosine Transforms (DCT) are computationally expensive due to their multiplicative complexity. This project implements several multiplierless 3D DCT approximations derived through high-order tensor theory. These approximations are suitable for energy-constrained systems and real-time applications like video coding and visual tracking.

## Key Features
* **Multiplierless Computation:** Implements approximations (e.g., MRDCT, LODCT) requiring only additions.
* **Tensor-Based Framework:** Uses $i$-mode product formalism to generalize 1D approximations into 3D and higher dimensions.
* **Visual Tracking:** Includes a discriminative tracker using 3D DCT subspace representation for efficient target search.
* **Modified Quantization:** A specialized procedure to absorb approximation scaling factors into the quantization volume.

## Project Organization
```
.
├── benchmarks
│   ├── compaction_benchmark.py   # 1D and 3D Energy compaction analysis
│   └── tracking_efficiency.py     # Tracker performance and PBM metrics
├── data                          # Video sequences (e.g., animal.mp4)
├── DESIGN.md                     # Mathematical foundation and roadmap
├── GEMINI.md                     # AI Context and development guidelines
├── src
│   └── coutinho2017
│       ├── core
│       │   ├── approximations.py # MRDCT/LODCT Matrix definitions
│       │   ├── quantization.py   # Modified quantization (Q*)
│       │   └── tensor_ops.py     # i-mode product engine
│       ├── tracking
│       │   └── tracker.py        # 3D DCT-based visual tracker
│       └── utils
│           ├── metrics.py        # PBM, PSNR, SSIM
│           └── video_io.py       # YUV/Video processing
├── tests
│   ├── test_3d_transform.py      # 3D Transform verification
│   └── test_compaction.py        # Energy compaction tests
└── pyproject.toml                # Project metadata and dependencies
```

## Getting Started

### 1. Installation
Install the project in editable mode with test dependencies:
```bash
pip install -e .[test]
```

### 2. Running Tests
Verify the mathematical integrity of the approximations:
```bash
pytest tests/
```

### 3. Running Benchmarks
Evaluate energy compaction and tracking efficiency:

**Energy Compaction (1D & 3D):**
```bash
python benchmarks/compaction_benchmark.py
```

**Tracking Efficiency:**
```bash
python benchmarks/tracking_efficiency.py
```
*Note: If `data/animal.mp4` is not present, the benchmark will automatically generate synthetic video data to evaluate the tracker's accuracy.*

## Implementation Details
The project is approximately **55% complete**. Currently supported features:
- **MRDCT (Modified Rounded DCT):** 14 additions, high energy compaction.
- **LODCT (Low-Order DCT):** Optimized for low-order coefficient accuracy.
- **3D i-mode Engine:** Successive application of 1D transforms across tensor modes.
- **Subspace Tracker:** Using 3D DCT coefficients as features for robust visual tracking (PBM metric implemented).
- **Modified Quantization:** Embedding scaling factors into the $Q$ volume to maintain multiplierless transforms.

**In Progress:**
- **BAS and CB-2011 Matrices:** Implementation of remaining Table I matrices.
- **Inverse 3D Transform:** Required for full video compression reproduction.
- **Formal Complexity Analysis:** Automated operation counting and verification.
