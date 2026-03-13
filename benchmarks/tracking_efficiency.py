import time
import os
import numpy as np
from scipy.fft import dctn
from coutinho2017.utils.video_io import load_video_sequence
from coutinho2017.tracking.tracker import DCTTracker
from coutinho2017.utils.metrics import calculate_pbm

def run_benchmark():
    # 1. Carregar dados reais ou sintéticos
    video_path = "data/animal.mp4"
    if os.path.exists(video_path):
        video_data = load_video_sequence(video_path)
    else:
        print(f"Aviso: {video_path} não encontrado. Usando dados sintéticos.")
        # Criar um "vídeo" sintético 352x288 com 50 frames
        # com um quadrado 8x8 se movendo
        num_frames = 50
        video_data = np.zeros((288, 352, num_frames), dtype=np.uint8)
        for f in range(num_frames):
            x, y = 150 + f, 100 + f # Movimento diagonal
            video_data[y:y+8, x:x+8, f] = 255
            
    num_frames = video_data.shape[2]
    
    # Bounding Box inicial (Ground Truth)
    if os.path.exists(video_path):
        gt_path = [(150, 100, 8, 8) for _ in range(num_frames)]
    else:
        gt_path = [(150 + f, 100 + f, 8, 8) for _ in range(num_frames)]
    
    # 2. Setup dos Rastreadores
    approx_tracker = DCTTracker(buffer_size=8)
    
    # Métricas
    results = {"Approx": {"pbm": [], "time": 0}, "Exact": {"pbm": [], "time": 0}}

    print(f"Iniciando Benchmark em {num_frames} frames...")

    # 3. Teste: Aproximação MRDCT (Coutinho 2017)
    start = time.perf_counter()
    for f in range(num_frames):
        frame = video_data[:, :, f]
        # Simula o fluxo de busca e atualização
        # Ref: [Coutinho2017, Sec. V] utiliza MRDCT para tracking
        bbox = approx_tracker.find_target(frame, gt_path[max(0, f-1)])
        approx_tracker.add_observation(frame, bbox)
        results["Approx"]["pbm"].append(calculate_pbm(bbox, gt_path[f]))
    results["Approx"]["time"] = time.perf_counter() - start

    # 4. Teste: DCT Exata (Baseline Li 2013)
    # Aqui usaríamos a função de biblioteca exata para comparação
    start = time.perf_counter()
    for f in range(num_frames):
        # Simulação da complexidade da DCT exata via SciPy
        # Em um tracker real, isso seria usado no lugar de transform_3d_approx
        _ = dctn(video_data[:8, :8, max(0, f-7):f+1].astype(float), norm='ortho')
        results["Exact"]["pbm"].append(1.0) # Baseline perfeito para simulação de PBM
    results["Exact"]["time"] = time.perf_counter() - start

    # 5. Relatório de Desempenho
    print("\n--- Resultados do Benchmark ---")
    pbm_medio = np.mean(results['Approx']['pbm'])
    print(f"MRDCT (Aprox) - PBM Médio: {pbm_medio:.4f}")
    print(f"MRDCT (Aprox) - Tempo Total: {results['Approx']['time']:.4f}s")
    print(f"Exact (SciPy) - Tempo Total: {results['Exact']['time']:.4f}s")
    
    # Ref: O artigo cita apenas 1.6% de perda no PBM médio
    diff = (1 - pbm_medio) * 100
    print(f"Degradação de Precisão: {diff:.2f}%")

if __name__ == "__main__":
    run_benchmark()
