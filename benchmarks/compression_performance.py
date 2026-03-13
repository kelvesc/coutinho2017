import numpy as np
from scipy.fft import dctn, idctn
from coutinho2017.core.tensor_ops import transform_3d_approx, inverse_transform_3d_approx, discard_coefficients
from coutinho2017.core.approximations import MRDCT, LODCT, BAS2008, CB2011
from coutinho2017.utils.metrics import calculate_psnr, calculate_ssim

def run_compression_benchmark():
    """
    Evaluates the compression performance of different DCT approximations.
    Simulates Section VI.B of Coutinho et al. 2017.
    """
    # 1. Create a synthetic 3D video cube (8x8x8)
    # Using a correlated signal model: x[i,j,k] = rho^(i+j+k)
    N = 8
    rho = 0.95
    video_cube = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                video_cube[i, j, k] = rho**(i + j + k)
    
    # Scale to 0-255 range
    video_cube = video_cube * 255.0

    methods = {
        "MRDCT": MRDCT(),
        "LODCT": LODCT(),
        "BAS-2008": BAS2008(),
        "CB-2011": CB2011()
    }

    # Fraction of coefficients to keep (m/N^3)
    keep_ratios = [0.1, 0.2, 0.4]
    
    print(f"\n{'METHOD':<12} | {'RATIO':<6} | {'PSNR (dB)':<10} | {'SSIM':<8}")
    print("-" * 45)

    # 2. Baseline: Exact DCT
    for r in keep_ratios:
        Y_exact = dctn(video_cube, norm='ortho')
        Y_thresh = discard_coefficients(Y_exact, r)
        reconstructed = idctn(Y_thresh, norm='ortho')
        reconstructed = np.clip(reconstructed, 0, 255)
        
        psnr = calculate_psnr(video_cube, reconstructed)
        ssim = calculate_ssim(video_cube, reconstructed)
        print(f"{'EXACT':<12} | {r:<6.2f} | {psnr:<10.2f} | {ssim:<8.4f}")
    
    print("-" * 45)

    # 3. Approximations
    for name, method in methods.items():
        for r in keep_ratios:
            # Forward Transform
            Y = transform_3d_approx(video_cube, method)
            
            # Compression (discard small coefficients)
            Y_comp = discard_coefficients(Y, r)
            
            # Inverse Transform
            reconstructed = inverse_transform_3d_approx(Y_comp, method)
            reconstructed = np.clip(reconstructed, 0, 255)
            
            psnr = calculate_psnr(video_cube, reconstructed)
            ssim = calculate_ssim(video_cube, reconstructed)
            print(f"{name:<12} | {r:<6.2f} | {psnr:<10.2f} | {ssim:<8.4f}")

if __name__ == "__main__":
    run_compression_benchmark()
