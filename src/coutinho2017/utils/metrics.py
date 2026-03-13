import numpy as np
from typing import Tuple

def calculate_pbm(
    tracked_bbox: Tuple[int, int, int, int], 
    ground_truth_bbox: Tuple[int, int, int, int]
) -> float:
    """
    Computes the Position-Based Measure (PBM).
    Value 1 indicates a perfect match; 0 indicates a failure.
    Ref: [cite: 513, 514, 515, 684]
    """
    # Unpack (x, y, w, h)
    xt, yt, wt, ht = tracked_bbox
    xg, yg, wg, hg = ground_truth_bbox
    
    # Calculate Centroids [cite: 683]
    cent_t = np.array([xt + wt/2, yt + ht/2])
    cent_g = np.array([xg + wg/2, yg + hg/2])
    
    # Threshold/Normalization factor (Th) [cite: 517]
    th = (wt + ht + wg + hg) / 2.0
    
    # Distance between centroids [cite: 518]
    distance = np.linalg.norm(cent_t - cent_g)
    
    # Check for intersection (simplified as non-zero area overlap)
    # Ref: [cite: 519, 520]
    has_intersection = not (
        xt + wt < xg or xg + wg < xt or 
        yt + ht < yg or yg + hg < yt
    )
    
    if not has_intersection:
        return 0.0
    
    # PBM formula: 1 - min(1, dist/Th) [cite: 514, 515, 518]
    pbm = 1.0 - min(1.0, distance / th)
    return float(pbm)

def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 255.0) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR).
    Ref: [cite: Section VI.B]
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes a simplified Structural Similarity Index (SSIM).
    Ref: [cite: Section VI.B]
    Note: This is a basic implementation of the SSIM formula.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

    return num / den