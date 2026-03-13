import numpy as np
import pytest
from coutinho2017.core.tensor_ops import transform_3d_approx, inverse_transform_3d_approx, discard_coefficients
from coutinho2017.core.approximations import MRDCT, BAS2008
from coutinho2017.utils.metrics import calculate_psnr

def test_reconstruction_no_discard():
    """Verifies that the transform and its inverse reconstruct the original signal (approximately)."""
    N = 8
    # Use a correlated signal instead of random noise
    rho = 0.95
    video_cube = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                video_cube[i, j, k] = rho**(i + j + k)
    video_cube *= 255.0
    
    method = BAS2008()
    
    # Forward
    Y = transform_3d_approx(video_cube, method)
    # Inverse
    reconstructed = inverse_transform_3d_approx(Y, method)
    reconstructed = np.clip(reconstructed, 0, 255)
    
    # PSNR should be reasonably high for BAS2008 with correlated data
    psnr = calculate_psnr(video_cube, reconstructed)
    assert psnr > 20  # BAS2008 is expected to be decent

def test_discard_coefficients_all():
    """Tests discarding all coefficients."""
    N = 8
    tensor = np.ones((N, N, N))
    compressed = discard_coefficients(tensor, 0.0)
    assert np.all(compressed == 0)

def test_discard_coefficients_none():
    """Tests discarding no coefficients."""
    N = 8
    tensor = np.ones((N, N, N))
    compressed = discard_coefficients(tensor, 1.0)
    assert np.all(compressed == tensor)

def test_discard_coefficients_fraction():
    """Tests discarding a fraction of coefficients."""
    N = 8
    tensor = np.arange(N**3).reshape((N, N, N))
    keep_ratio = 0.5
    num_elements = N**3
    expected_keep = int(num_elements * keep_ratio)
    
    compressed = discard_coefficients(tensor, keep_ratio)
    num_nonzero = np.count_nonzero(compressed)
    
    # Depending on values, it might keep slightly more if there are ties, 
    # but for unique values it should be exact.
    assert num_nonzero == expected_keep
