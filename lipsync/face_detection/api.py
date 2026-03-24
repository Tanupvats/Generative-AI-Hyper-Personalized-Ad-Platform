import torch
from enum import Enum
import numpy as np

# A streamlined version of the FaceAlignment class built purely for our API
class LandmarksType(Enum):
    _2D = 1
    _2halfD = 2
    _3D = 3

class FaceAlignment:
    def __init__(self, landmarks_type, flip_input=False, device='cuda'):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Lazily load the Heavy S3FD detector to prevent overhead issues
        from .detection.sfd_detector import SFDDetector
        self.face_detector = SFDDetector(device=device)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1] # BGR to RGB
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)
            
            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))

        return results