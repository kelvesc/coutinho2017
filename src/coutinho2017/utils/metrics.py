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