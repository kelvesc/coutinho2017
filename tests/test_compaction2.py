import numpy as np
from scipy.fft import dct
from src.core.approximations import mrdct_8pt, lodct_8pt
# from typing import Dict

def run_benchmarks() -> None:
    rho = 0.95
    x = np.array([rho**n for n in range(8)])
    
    # Exact Baseline
    X_exact = dct(x, type=2, norm='ortho')
    energy_exact = np.sum(X_exact**2)
    
    methods = {
        "MRDCT": mrdct_8pt(x),
        "LODCT": lodct_8pt(x)
    }
    
    print(f"{'Method':<10} | {'DC Energy %':<12} | {'Top-2 Compaction %':<18}")
    print("-" * 45)
    
    # Print Exact Baseline
    dc_exact = (X_exact[0]**2 / energy_exact) * 100
    comp2_exact = (np.sum(X_exact[:2]**2) / energy_exact) * 100
    print(f"{'Exact':<10} | {dc_exact:<12.2f} | {comp2_exact:<18.2f}")

    for name, X_approx in methods.items():
        energy_approx = np.sum(X_approx**2)
        dc_perc = (X_approx[0]**2 / energy_approx) * 100
        comp2_perc = (np.sum(X_approx[:2]**2) / energy_approx) * 100
        
        print(f"{name:<10} | {dc_perc:<12.2f} | {comp2_perc:<18.2f}")

if __name__ == "__main__":
    run_benchmarks()