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

def get_bas2008_T8() -> np.ndarray:
    """Returns the integer matrix T8 for the BAS-2008 approximation."""
    # Row elements as per Table I of Coutinho 2017 (Ref [44])
    # Note: Multiplierless implementation uses 1/2 as bit shifts
    return np.array([
        [ 1,  1,  1,  1,  1,  1,  1,  1],
        [ 1,  0.5, 0,  0,  0,  0, -0.5, -1],
        [ 1,  0,  0, -1, -1,  0,  0,  1],
        [ 0,  1, -0.5, 0,  0,  0.5, -1,  0],
        [ 1, -1, -1,  1,  1, -1, -1,  1],
        [ 0.5, -1, 0,  0,  0,  0,  1, -0.5],
        [ 0,  0.5, 0, -1,  1,  0, -0.5,  0],
        [ 0,  0,  1,  0,  0, -1,  0,  0]
    ], dtype=float)

def get_bas2008_S8() -> np.ndarray:
    """Returns the diagonal scaling matrix S8 for BAS-2008."""
    # S_ii = 1 / sqrt(sum(T_ij^2))
    diag_vals = [
        1/np.sqrt(8), 1/np.sqrt(2.5), 1/2, 1/np.sqrt(2.5),
        1/np.sqrt(8), 1/np.sqrt(2.5), 1/np.sqrt(2.5), 1/np.sqrt(2)
    ]
    return np.diag(diag_vals)

class BAS2008:
    """BAS-2008 approximation class."""
    def get_T8(self) -> np.ndarray: return get_bas2008_T8()
    def get_S8(self) -> np.ndarray: return get_bas2008_S8()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return get_bas2008_S8() @ (get_bas2008_T8() @ x)

def get_cb2011_T8() -> np.ndarray:
    """Returns the integer matrix T8 for the CB-2011 (RDCT) approximation."""
    # Ref: [42] Cintra and Bayer, 2011.
    return np.array([
        [ 1,  1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  0,  0, -1, -1, -1],
        [ 1,  0, -1, -1, -1, -1,  0,  1],
        [ 0,  1, -1,  0,  0,  1, -1,  0],
        [ 1, -1, -1,  1,  1, -1, -1,  1],
        [ 1, -1,  0,  0,  0,  0,  1, -1],
        [ 0,  1,  0, -1,  1,  0, -1,  0],
        [ 1,  0, -1,  0,  0,  1,  0, -1]
    ], dtype=float)

def get_cb2011_S8() -> np.ndarray:
    """Returns the diagonal scaling matrix S8 for CB-2011."""
    diag_vals = [
        1/np.sqrt(8), 1/np.sqrt(6), 1/np.sqrt(6), 1/2,
        1/np.sqrt(8), 1/2, 1/2, 1/2
    ]
    return np.diag(diag_vals)

class CB2011:
    """CB-2011 (RDCT) approximation class."""
    def get_T8(self) -> np.ndarray: return get_cb2011_T8()
    def get_S8(self) -> np.ndarray: return get_cb2011_S8()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return get_cb2011_S8() @ (get_cb2011_T8() @ x)
