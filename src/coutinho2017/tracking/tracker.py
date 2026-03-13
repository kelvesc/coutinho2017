import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.fft import dct
from coutinho2017.core.tensor_ops import i_mode_product, transform_3d_approx
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
        self.target_features: Optional[np.ndarray] = None
        
    def add_observation(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """
        Extracts a target patch and adds it to the temporal buffer.
        """
        x, y, w, h = bbox
        # Boundary safety
        x_min, y_min = max(0, x), max(0, y)
        x_max, y_max = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
        patch = frame[y_min:y_max, x_min:x_max]
        
        # Resize to match expected N1 x N2 (8x8)
        resized_patch = cv2.resize(patch, self.target_sz, interpolation=cv2.INTER_AREA)
        
        self.buffer.append(resized_patch)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        # Update our reference target model whenever we add a confirmed observation
        self.target_features = self._compute_features(self.buffer)

    def _compute_features(self, buffer_subset: List[np.ndarray]) -> np.ndarray:
        """
        Computes the 3D DCT features for a given buffer.
        Ref: Section V - Visual Tracking (page 2305)
        """
        n3 = len(buffer_subset)
        cube = np.stack(buffer_subset, axis=-1).astype(np.float32)
        
        # Spatial dimensions (N1, N2) always use the approximation (MRDCT)
        # Ref: "We utilized the MRDCT matrix for C1 and C2"
        t8_spatial = self.approx.get_T8()
        s8_spatial = self.approx.get_S8()
        c_spatial = s8_spatial @ t8_spatial
        
        res = i_mode_product(cube, c_spatial, mode=1)
        res = i_mode_product(res, c_spatial, mode=2)
        
        # Temporal dimension (N3) logic:
        if n3 < self.buffer_size:
            # Ref: "exact DCT matrix CN3 instead of CN3 until N3 reaches 8"
            # Apply exact DCT across the temporal axis (axis=2)
            features = dct(res, axis=2, type=2, norm='ortho')
        else:
            # Ref: "Then, we computed the 3D MRDCT approximation... for all remaining frames"
            # Apply the approximate MRDCT across the temporal axis
            features = i_mode_product(res, c_spatial, mode=3)
            
        return features

    def find_target(self, frame: np.ndarray, last_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Searches for the target in the current frame based on the 3D DCT subspace.
        """
        if self.target_features is None:
            return last_bbox

        x_last, y_last, w, h = last_bbox
        search_radius = 16  # Pixels to search around the last position
        best_score = float('inf')
        best_bbox = last_bbox

        # 1. Define candidate search space (sliding window)
        # Optimization: step=2 for efficiency as per common tracker implementations
        for dx in range(-search_radius, search_radius + 1, 2):
            for dy in range(-search_radius, search_radius + 1, 2):
                x_cand, y_cand = x_last + dx, y_last + dy
                
                # Boundary check
                if x_cand < 0 or y_cand < 0 or x_cand + w > frame.shape[1] or y_cand + h > frame.shape[0]:
                    continue
                
                # 2. Extract candidate and form a temporary cube
                cand_patch = frame[y_cand:y_cand+h, x_cand:x_cand+w]
                cand_resized = cv2.resize(cand_patch, self.target_sz, interpolation=cv2.INTER_AREA)
                
                # We simulate the buffer if this candidate were accepted
                temp_buffer = (self.buffer[1:] if len(self.buffer) == self.buffer_size else self.buffer) + [cand_resized]
                cand_features = self._compute_features(temp_buffer)
                
                # 3. Calculate distance (L2 norm) in the DCT domain
                # Ref: Subspace tracking via feature matching
                score = np.linalg.norm(self.target_features - cand_features)
                
                if score < best_score:
                    best_score = score
                    best_bbox = (x_cand, y_cand, w, h)
                        
        return best_bbox