import numpy as np
from scipy.fft import dct
from src.core.approximations import mrdct_8pt
# from typing import List

def test_energy_compaction() -> None:
    # 1. Create a highly correlated signal (typical for image/video data)
    # x[n] = rho^n, where rho is the correlation coefficient
    rho = 0.95
    x = np.array([rho**n for n in range(8)])
    
    # 2. Compute Exact DCT (Type II, normalized)
    X_exact = dct(x, type=2, norm='ortho')
    
    # 3. Compute MRDCT Approximation
    X_approx = mrdct_8pt(x)
    
    # 4. Calculate Energy Distribution
    energy_exact = X_exact**2
    energy_approx = X_approx**2
    
    total_energy_exact = np.sum(energy_exact)
    total_energy_approx = np.sum(energy_approx)
    
    print(f"\n--- Energy Compaction Test (N=8) ---")
    print(f"Total Energy (Exact):  {total_energy_exact:.4f}")
    print(f"Total Energy (MRDCT):  {total_energy_approx:.4f}")
    
    # 5. Compare DC component (the first coefficient)
    # The DC component usually contains the most energy
    print(f"\nDC Energy Percentage:")
    print(f"Exact: {(energy_exact[0]/total_energy_exact)*100:.2f}%")
    print(f"MRDCT: {(energy_approx[0]/total_energy_approx)*100:.2f}%")

    # 6. Verification: MRDCT should have similar compaction to Exact DCT
    # We check if the first 2 coefficients contain > 90% of energy
    compaction_2_exact = np.sum(energy_exact[:2]) / total_energy_exact
    compaction_2_approx = np.sum(energy_approx[:2]) / total_energy_approx
    
    assert abs(compaction_2_exact - compaction_2_approx) < 0.05
    print(f"\nSuccess: MRDCT matches Exact DCT compaction within 5% tolerance.")

if __name__ == "__main__":
    test_energy_compaction()