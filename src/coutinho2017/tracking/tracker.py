import numpy as np
import cv2
from typing import List, Tuple, Optional
from coutinho2017.core.tensor_ops import transform_3d_approx
from coutinho2017.core.approximations import MRDCT

class DCTTracker:
    def __init__(self, buffer_size: int = 8, target_sz: Tuple[int, int] = (8, 8)):
        """
        Initializes the 3D DCT-based tracker.
        
        Args:
            buffer_size: Max number of frames (N3) in the temporal buffer.
            target_sz: Spatial dimensions (N1, N2) for the target patch.
        """
        self.buffer_size = buffer_size
        self.target_sz = target_sz
        self.buffer: List[np.ndarray] = []
        self.approx = MRDCT()
        
    def add_observation(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """
        Extracts a target patch and adds it to the temporal buffer.
        """
        x, y, w, h = bbox
        patch = frame[y:y+h, x:x+w]
        # Resize to match expected N1 x N2 (8x8)
        resized_patch = cv2.resize(patch, self.target_sz, interpolation=cv2.INTER_AREA)
        
        self.buffer.append(resized_patch)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def get_subspace_features(self) -> Optional[np.ndarray]:
        """
        Computes the 3D DCT features for the current buffer.
        If the buffer is full (N3=8), it applies the multiplierless 3D MRDCT.
        Ref: 
        """
        if len(self.buffer) < self.buffer_size:
            return None
            
        # Stack frames into a 3D Tensor (N1, N2, N3)
        cube = np.stack(self.buffer, axis=-1).astype(np.float32)
        
        # Apply the 3D Approximation (multiplierless engine)
        features = transform_3d_approx(cube, self.approx)
        return features
    
    def find_target(self, frame: np.ndarray, last_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Searches for the target in the current frame based on the 3D DCT subspace.
        
        Args:
            frame: The current video frame.
            last_bbox: (x, y, w, h) of the target in the previous frame.
        """
        x_last, y_last, w, h = last_bbox
        search_radius = 20  # Pixels to search around the last position
        best_score = float('inf')
        best_bbox = last_bbox

        # 1. Define candidate search space (simplified sliding window)
        for dx in range(-search_radius, search_radius + 1, 2):
            for dy in range(-search_radius, search_radius + 1, 2):
                x_cand, y_cand = x_last + dx, y_last + dy
                
                # Boundary check
                if x_cand < 0 or y_cand < 0 or x_cand + w > frame.shape[1] or y_cand + h > frame.shape[0]:
                    continue
                
                # 2. Extract and process candidate
                cand_patch = frame[y_cand:y_cand+h, x_cand:x_cand+w]
                cand_resized = cv2.resize(cand_patch, self.target_sz, interpolation=cv2.INTER_AREA)
                
                # To compare, we need a 3D view. We temporarily use the 
                # candidate as the newest frame in the temporal buffer.
                # Ref: 3D DCT compact representations 
                temp_buffer = self.buffer[1:] + [cand_resized]
                cube = np.stack(temp_buffer, axis=-1).astype(np.float32)
                cand_features = transform_3d_approx(cube, self.approx)
                
                # 3. Calculate distance to our target model
                # (Assuming the DC and low-frequency components hold the identity)
                target_model = self.get_subspace_features()
                if target_model is not None:
                    score = np.linalg.norm(target_model - cand_features)
                    
                    if score < best_score:
                        best_score = score
                        best_bbox = (x_cand, y_cand, w, h)
                        
        return best_bbox