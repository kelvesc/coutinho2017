import numpy as np
from scipy.fft import dct
from coutinho2017.core.approximations import MRDCT, LODCT
from typing import Dict, Type

def run_benchmarks() -> None:
    # 1. Sinal de entrada altamente correlacionado (rho = 0.95)
    # Ref: [Coutinho2017] utiliza este modelo para aproximar a KLT
    rho = 0.95
    x = np.array([rho**n for n in range(8)], dtype=float)
    
    # 2. Baseline: DCT Exata (SciPy Tipo-II ortonormal)
    X_exact = dct(x, type=2, norm='ortho')
    energy_exact = np.sum(X_exact**2)
    
    # 3. Métodos de Aproximação (Coutinho 2017)
    # Instanciamos as classes que seguem o Protocol Approximation
    methods = {
        "MRDCT": MRDCT(),
        "LODCT": LODCT()
    }
    
    print(f"\n{'MÉTODO':<12} | {'ENERGIA DC %':<15} | {'COMPACTAÇÃO TOP-2 %':<20}")
    print("-" * 55)
    
    # Cálculo para a DCT Exata
    dc_exact = (X_exact[0]**2 / energy_exact) * 100
    comp2_exact = (np.sum(X_exact[:2]**2) / energy_exact) * 100
    print(f"{'EXATA':<12} | {dc_exact:<15.2f} | {comp2_exact:<20.2f}")

    # Cálculo para as Aproximações
    for name, method in methods.items():
        # Execução via matrizes T8 e S8 conforme o Protocol
        T8 = method.get_T8()
        S8 = method.get_S8()
        X_approx = S8 @ (T8 @ x)
        
        energy_approx = np.sum(X_approx**2)
        dc_perc = (X_approx[0]**2 / energy_approx) * 100
        comp2_perc = (np.sum(X_approx[:2]**2) / energy_approx) * 100
        
        print(f"{name:<12} | {dc_perc:<15.2f} | {comp2_perc:<20.2f}")

if __name__ == "__main__":
    run_benchmarks()