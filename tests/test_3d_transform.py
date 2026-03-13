import numpy as np
from coutinho2017.core.tensor_ops import transform_3d_approx
from coutinho2017.core.approximations import MRDCT

def test_3d_output_shape() -> None:
    # Create an 8x8x8 'video cube'
    cube = np.random.rand(8, 8, 8)
    
    # Run the 3D MRDCT
    transformed = transform_3d_approx(cube, MRDCT())
    
    # Verify dimensions are preserved
    assert transformed.shape == (8, 8, 8)