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

def get_mrdct_S8() -> np.ndarray:
    """Returns the diagonal scaling matrix S8 for MRDCT[cite: 179]."""
    # diag(1/sqrt(8), 1/sqrt(2), 1/2, 1/sqrt(2), 1/sqrt(8), 1/sqrt(2), 1/2, 1/sqrt(2))
    diag_vals = [
        1/np.sqrt(8), 1/np.sqrt(2), 1/2, 1/np.sqrt(2),
        1/np.sqrt(8), 1/np.sqrt(2), 1/2, 1/np.sqrt(2)
    ]
    return np.diag(diag_vals)

def mrdct_8pt(x: np.ndarray) -> np.ndarray:
    """Computes the 8-point MRDCT: X = S8 * T8 * x[cite: 159]."""
    T8 = get_mrdct_T8()
    S8 = get_mrdct_S8()
    return S8 @ (T8 @ x)


def get_lodct_T8() -> np.ndarray:
    """
    Returns the integer matrix T8 for the LODCT approximation.
    Ref: [cite: 179] (Coutinho 2017, Table I)
    """
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
    """
    Returns the diagonal scaling matrix S8 for LODCT.
    Ref: [cite: 179] (Coutinho 2017, Table I)
    """
    # diag(1/sqrt(8), 1/sqrt(6), 1/sqrt(5), 1/sqrt(6), 1/sqrt(8), 1/sqrt(6), 1/sqrt(5), 1/sqrt(6))
    diag_vals = [
        1/np.sqrt(8), 1/np.sqrt(6), 1/np.sqrt(5), 1/np.sqrt(6),
        1/np.sqrt(8), 1/np.sqrt(6), 1/np.sqrt(5), 1/np.sqrt(6)
    ]
    return np.diag(diag_vals)

def lodct_8pt(x: np.ndarray) -> np.ndarray:
    """Computes the 8-point LODCT: X = S8 * T8 * x."""
    T8 = get_lodct_T8()
    S8 = get_lodct_S8()
    return S8 @ (T8 @ x)