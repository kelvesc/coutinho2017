import numpy as np
from typing import Optional

def generate_modified_q_volume(
    base_q_volume: np.ndarray, 
    d_vector: np.ndarray
) -> np.ndarray:
    """
    Generates the modified quantization volume Q*.
    
    This embeds the diagonal scaling factors (d) into the quantization 
    step so the 3D transform can remain multiplierless.
    
    Formula: q*[k1, k2, k3] = q[k1, k2, k3] / (d[k1] * d[k2] * d[k3])
    Ref: 
    """
    # Create the 3D denominator using an outer product (Kronecker-like)
    # d_3d[k1, k2, k3] = d[k1] * d[k2] * d[k3]
    d_3d = np.einsum('i,j,k->ijk', d_vector, d_vector, d_vector)
    
    # Return the modified volume
    return base_q_volume / d_3d

def quantize_3d(
    transformed_tensor: np.ndarray, 
    modified_q_volume: np.ndarray
) -> np.ndarray:
    """
    Applies the modified quantization step.
    Ref: [cite: 346]
    """
    return np.round(transformed_tensor / modified_q_volume).astype(np.int32)

def dequantize_3d(
    quantized_tensor: np.ndarray,
    modified_q_volume: np.ndarray
) -> np.ndarray:
    """
    Reverses the quantization step.
    Ref: [cite: 346]
    """
    return (quantized_tensor * modified_q_volume).astype(np.float64)

def generate_base_3d_q_volume(N: int = 8, quality: float = 50.0) -> np.ndarray:
    """
    Generates a basic 3D quantization volume based on Euclidean distance.
    Higher 'quality' values result in smaller quantization steps.
    """
    # Create coordinate grids
    k1, k2, k3 = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
    
    # Distance from DC coefficient (0,0,0)
    # Higher frequencies (further from origin) get larger quantization steps
    dist = k1 + k2 + k3
    
    # Simple linear scaling model for Q
    # This is a placeholder for the specialized volumes in Ref [71]
    scale = (100.0 - quality) / 50.0
    q_volume = 1 + (1 + dist) * scale
    
    return q_volume.astype(np.float64)