# TODO: Reproduction of Coutinho 2017

Progress tracking for the implementation of *"Low-Complexity Multidimensional DCT Approximations for High-Order Tensor Data Decorrelation"*.

## Phase 1: Core Framework [75%]
- [x] **1D Multiplierless Matrices ($T_8$):**
    - [x] MRDCT Implementation
    - [x] LODCT Implementation
    - [ ] BAS-2008 Implementation (Table I)
    - [ ] CB-2011 Implementation (Table I)
- [x] **Tensor Product Engine:**
    - [x] $i$-mode product implementation (`tensor_ops.py`)
    - [x] 3D Transform wrapper
- [x] **Modified Quantization Volume ($Q^*$):**
    - [x] $Q^*$ generation embedding scaling factors $d$
    - [x] 3D Quantization logic

## Phase 2: Visual Tracking (Section V) [50%]
- [x] **Feature Extraction:**
    - [x] 3D DCT Subspace Representation
- [x] **Target Search:**
    - [x] Sliding window candidate evaluation
- [ ] **Incremental Logic:**
    - [ ] Temporal buffer rolling window (8 frames) - *Basic implementation in tracker.py*
    - [ ] Subspace update strategy as per paper
- [x] **Metrics:**
    - [x] PBM (Position-Based Measure) implementation

## Phase 3: Verification & Reproduction (Section VI) [30%]
- [x] **Energy Compaction:**
    - [x] 1D Compaction analysis (MRDCT vs Exact)
    - [ ] 3D Compaction analysis (rudimentary implementation in tests)
- [ ] **Compression Performance:**
    - [ ] Implement Inverse 3D Transform (Approximate)
    - [ ] Coefficient discarding/thresholding pipeline
    - [ ] PSNR and SSIM comparison on CIF sequences (Foreman, Mother-daughter)
- [ ] **Complexity Analysis:**
    - [ ] Formal count of additions vs multiplications
    - [ ] Verification of the "zero-multiplication" property in the 3D pipeline

## Phase 4: Infrastructure
- [x] Standardize imports and package structure
- [x] Setup `pytest` verification suite
- [ ] Create data downloader for standard CIF sequences
- [ ] Script to reproduce Table III and Figure 5 from the paper
