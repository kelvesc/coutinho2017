import numpy as np
# from typing import Tuple

def get_mrdct_T8() -> np.ndarray:
    """Returns the integer matrix T8 for the MRDCT approximation[cite: 179]."""
    return np.array([
        [1,  1,  1,  1,  1,  1,  1,  1],
        [1,  1,  1,  0,  0, -1, -1, -1],
        [1,  0, -1, -1, -1, -1,  0,  1],
        [1,  0, -1,  0,  0,  1,  0, -1],
        [1, -1, -1,  1,  1, -1, -1,  1],
        [0, -1,  0,  1, -1,  0,  1,  0],
        [0,  1, -1,  0,  0,  1, -1,  0],
        [0,  1,  0, -1,  1,  0, -1,  0]
    ], dtype=float)
