import numpy as np
from src.core.tensor_ops import transform_3d_approx
from src.core.approximations import mrdct_8pt, get_mrdct_T8, get_mrdct_S8

class MRDCT_Wrapper:
    """Wrapper to satisfy the Approximation protocol."""
    def get_T8(self): return get_mrdct_T8()
    def get_S8(self): return get_mrdct_S8()

def test_3d_output_shape() -> None:
    # Create an 8x8x8 'video cube'
    cube = np.random.rand(8, 8, 8)
    
    # Run the 3D MRDCT
    transformed = transform_3d_approx(cube, MRDCT_Wrapper())
    
    # Verify dimensions are preserved [cite: 198]
    assert transformed.shape == (8, 8, 8)
    print("Success: 3D Transform preserved tensor dimensions.")

if __name__ == "__main__":
    test_3d_output_shape()