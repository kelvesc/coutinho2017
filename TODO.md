# TODO: Reproduction of Coutinho 2017

Progress tracking for the implementation of *"Low-Complexity Multidimensional DCT Approximations for High-Order Tensor Data Decorrelation"*.

## Phase 1: Core Framework [100%]
- [x] **1D Multiplierless Matrices ($T_8$):**
    - [x] MRDCT Implementation
    - [x] LODCT Implementation
    - [x] BAS-2008 Implementation (Table I)
    - [x] CB-2011 Implementation (Table I)
- [x] **Tensor Product Engine:**
    - [x] $i$-mode product implementation (`tensor_ops.py`)
    - [x] 3D Transform wrapper
- [x] **Modified Quantization Volume ($Q^*$):**
    - [x] $Q^*$ generation embedding scaling factors $d$
    - [x] 3D Quantization logic

## Phase 2: Visual Tracking (Section V) [100%]
- [x] **Feature Extraction:**
    - [x] 3D DCT Subspace Representation
- [x] **Target Search:**
    - [x] Sliding window candidate evaluation
- [x] **Incremental Logic:**
    - [x] Temporal buffer rolling window (8 frames)
    - [x] Subspace update strategy as per paper
- [x] **Metrics:**
    - [x] PBM (Position-Based Measure) implementation

## Phase 3: Verification & Reproduction (Section VI) [70%]
- [x] **Energy Compaction:**
    - [x] 1D Compaction analysis (MRDCT vs Exact)
    - [x] 3D Compaction analysis (rudimentary implementation in tests)
- [x] **Compression Performance:**
    - [x] Implement Inverse 3D Transform (Approximate)
    - [x] Coefficient discarding/thresholding pipeline
    - [x] PSNR and SSIM comparison on synthetic correlated data
- [ ] **Complexity Analysis:**
    - [ ] Formal count of additions vs multiplications
    - [ ] Verification of the "zero-multiplication" property in the 3D pipeline

## Phase 4: Infrastructure
- [x] Standardize imports and package structure
- [x] Setup `pytest` verification suite
- [x] Create data downloader for standard CIF sequences
- [ ] Script to reproduce Table III and Figure 5 from the paper
