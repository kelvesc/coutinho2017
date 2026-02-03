import numpy as np
from src.core.tensor_ops import i_mode_product
from src.core.approximations import get_lodct_T8, get_lodct_S8
from src.core.quantization import generate_base_3d_q_volume, generate_modified_q_volume

def run_pipeline() -> None:
    # 1. Setup: Create a dummy 8x8x8 video block (tensor T)
    # Using a gradient to simulate spatial/temporal correlation
    N = 8
    grid = np.linspace(0, 255, N)
    T = np.meshgrid(grid, grid, grid, indexing='ij')[0] + np.random.normal(0, 1, (N, N, N))
    
    # 2. Preparation: LODCT Matrices and Quantization
    T8 = get_lodct_T8()
    S8 = get_lodct_S8()
    d_vector = np.diag(S8)
    
    base_Q = generate_base_3d_q_volume(N=N, quality=70.0)
    Q_star_quant = generate_modified_q_volume(base_Q, d_vector)
    # For dequantization: q_inv[k1,k2,k3] = q[k1,k2,k3] * d[k1]*d[k2]*d[k3] (Eq. 36)
    d_3d = np.einsum('i,j,k->ijk', d_vector, d_vector, d_vector)
    Q_star_dequant = base_Q * d_3d

    # 3. Encoding Flow
    # Step A: Multiplierless 3D Transform (T x1 T8 x2 T8 x3 T8)
    A = i_mode_product(T, T8, mode=1)
    A = i_mode_product(A, T8, mode=2)
    A = i_mode_product(A, T8, mode=3)
    
    # Step B: Modified Quantization (Eq. 35)
    Y_quant = np.round(A / Q_star_quant).astype(np.int32)

    # 4. Decoding Flow
    # Step C: Modified Dequantization (Eq. 36)
    Y_hat = Y_quant * Q_star_dequant
    
    # Step D: Inverse Transform (Multiplierless)
    # For LODCT, we use the transpose of T8 (assuming quasi-orthogonality)
    T8_inv = T8.T
    T_hat = i_mode_product(Y_hat, T8_inv, mode=1)
    T_hat = i_mode_product(T_hat, T8_inv, mode=2)
    T_hat = i_mode_product(T_hat, T8_inv, mode=3)

    # 5. Performance Metrics (Eq. 38)
    mse = np.mean((T - T_hat)**2)
    print(f"--- Pipeline Results (LODCT @ Quality 70) ---")
    print(f"Reconstruction MSE: {mse:.4f}")
    
    # A small MSE indicates successful energy preservation
    if mse < 50:
        print("Success: Low reconstruction error achieved.")

if __name__ == "__main__":
    run_pipeline()