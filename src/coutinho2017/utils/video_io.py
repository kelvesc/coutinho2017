import cv2
import numpy as np
from typing import Generator, Tuple

def load_video_sequence(
    video_path: str, 
    target_size: Tuple[int, int] = (352, 288), 
    max_frames: int = 296
) -> np.ndarray:
    """
    Carrega uma sequência de vídeo e a converte em um tensor 3D (YUV - Canal Y).
    Ref: 
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
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]
        
        # Redimensionar para o padrão CIF se necessário 
        if y_channel.shape[::-1] != target_size:
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
    Ref: 
    """
    h, w, t = video_tensor.shape
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            for k in range(0, t - block_size + 1, block_size):
                yield video_tensor[i:i+block_size, j:j+block_size, k:k+block_size]