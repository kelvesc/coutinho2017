import numpy as np
from typing import List, Protocol

class Approximation(Protocol):
    """Protocol for approximation methods like MRDCT or LODCT."""
    def __call__(self, x: np.ndarray) -> np.ndarray: ...
    def get_T8(self) -> np.ndarray: ...
    def get_S8(self) -> np.ndarray: ...

def i_mode_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """
    Computes the i-mode product of a tensor by a matrix.
    Ref: [cite: 121, 122]
    """
    # Aligning mode with 0-based indexing (Paper uses 1, 2, 3)
    mode_idx = mode - 1
    
    # Use tensordot to sum over the specified axis
    res = np.tensordot(matrix, tensor, axes=(1, mode_idx))
    
    # Roll the new dimension back to the original position
    return np.moveaxis(res, 0, mode_idx)

def transform_3d_approx(tensor: np.ndarray, method: Approximation) -> np.ndarray:
    """
    Computes the 3D Approximate DCT.
    Y = T x1 (SN*TN) x2 (SN*TN) x3 (SN*TN)
    Ref: [cite: 199, 216]
    """
    T8 = method.get_T8()
    S8 = method.get_S8()
    
    # Step 1: Apply low-complexity T8 to all 3 dimensions
    # This is the multiplierless part
    res = i_mode_product(tensor, T8, mode=1)
    res = i_mode_product(res, T8, mode=2)
    res = i_mode_product(res, T8, mode=3)
    
    # Step 2: Apply scaling matrices S8
    # In a real codec, these are merged into quantization [cite: 202, 326]
    res = i_mode_product(res, S8, mode=1)
    res = i_mode_product(res, S8, mode=2)
    res = i_mode_product(res, S8, mode=3)
    
    return res