import numpy as np
# from typing import Tuple

def get_mrdct_T8() -> np.ndarray:
    """Returns the integer matrix T8 for the MRDCT approximation."""
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

def get_mrdct_S8() -> np.ndarray:
    """Returns the diagonal scaling matrix S8 for MRDCT based on row norms."""
    # S_ii = 1 / sqrt(sum(T_ij^2))
    diag_vals = [
        1/np.sqrt(8), 1/np.sqrt(6), 1/np.sqrt(6), 1/2,
        1/np.sqrt(8), 1/2, 1/2, 1/2
    ]
    return np.diag(diag_vals)

def mrdct_8pt(x: np.ndarray) -> np.ndarray:
    """Computes the 8-point MRDCT: X = S8 * T8 * x."""
    T8 = get_mrdct_T8()
    S8 = get_mrdct_S8()
    return S8 @ (T8 @ x)


def get_lodct_T8() -> np.ndarray:
    """Returns the integer matrix T8 for the LODCT approximation."""
    return np.array([
        [ 1,  1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  0,  0,  0,  0, -1, -1],
        [ 1,  0, -1, -1, -1, -1,  0,  1],
        [ 0,  1, -1,  0,  0,  1, -1,  0],
        [ 1, -1, -1,  1,  1, -1, -1,  1],
        [ 1, -1,  0,  0,  0,  0,  1, -1],
        [ 0,  1,  0, -1,  1,  0, -1,  0],
        [ 1,  0, -1,  0,  0,  1,  0, -1]
    ], dtype=float)

def get_lodct_S8() -> np.ndarray:
    """Returns the diagonal scaling matrix S8 for LODCT based on row norms."""
    diag_vals = [
        1/np.sqrt(8), 1/2, 1/np.sqrt(6), 1/2,
        1/np.sqrt(8), 1/2, 1/2, 1/2
    ]
    return np.diag(diag_vals)

def lodct_8pt(x: np.ndarray) -> np.ndarray:
    """Computes the 8-point LODCT: X = S8 * T8 * x."""
    T8 = get_lodct_T8()
    S8 = get_lodct_S8()
    return S8 @ (T8 @ x)

class MRDCT:
    """MRDCT approximation class."""
    def get_T8(self) -> np.ndarray: return get_mrdct_T8()
    def get_S8(self) -> np.ndarray: return get_mrdct_S8()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return mrdct_8pt(x)

class LODCT:
    """LODCT approximation class."""
    def get_T8(self) -> np.ndarray: return get_lodct_T8()
    def get_S8(self) -> np.ndarray: return get_lodct_S8()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return lodct_8pt(x)
