import cv2
import os
import urllib.request
import numpy as np
from typing import Generator, Tuple, List

def load_video_sequence(
    video_path: str, 
    target_size: Tuple[int, int] = (352, 288), 
    max_frames: int = 296
) -> np.ndarray:
    """
    Carrega uma sequência de vídeo e a converte em um tensor 3D (YUV - Canal Y).
    Ref: [cite: 588, 591]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Converter para YUV e extrair apenas o canal de Luminância (Y)
        # O processamento de DCT 3D no artigo foca na luminância.
        y_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]
        
        # Redimensionar para o padrão CIF se necessário 
        if (y_channel.shape[1], y_channel.shape[0]) != target_size:
            y_channel = cv2.resize(y_channel, target_size)
            
        frames.append(y_channel)
        count += 1
        
    cap.release()
    # Retorna tensor no formato (Altura, Largura, Tempo) 
    return np.stack(frames, axis=-1)

def get_video_blocks(
    video_tensor: np.ndarray, 
    block_size: int = 8
) -> Generator[np.ndarray, None, None]:
    """
    Gera 'cubos' de pixels 8x8x8 a partir do tensor de vídeo.
    Ref: [cite: 585, 592]
    """
    h, w, t = video_tensor.shape
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            for k in range(0, t - block_size + 1, block_size):
                yield video_tensor[i:i+block_size, j:j+block_size, k:k+block_size]

def download_cif_sequences(data_dir: str = "data") -> List[str]:
    """
    Downloads standard CIF sequences for reproduction (Foreman, Mother-daughter).
    Ref: [cite: 589]
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # URLs from Xiph.org or similar open repositories
    sequences = {
        "foreman.mp4": "https://media.xiph.org/video/derf/mp4/foreman_cif.mp4",
        "mother-daughter.mp4": "https://media.xiph.org/video/derf/mp4/mother-daughter_cif.mp4"
    }
    
    downloaded_paths = []
    for name, url in sequences.items():
        dest_path = os.path.join(data_dir, name)
        if not os.path.exists(dest_path):
            print(f"Downloading {name}...")
            try:
                urllib.request.urlretrieve(url, dest_path)
                downloaded_paths.append(dest_path)
            except Exception as e:
                print(f"Error downloading {name}: {e}")
        else:
            downloaded_paths.append(dest_path)
            
    return downloaded_paths