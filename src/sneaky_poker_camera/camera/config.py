from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class CameraConfig:
    """Create the camera configuration parameters"""
    resolution = (2592, 1944)
    focus_sweep_steps = 10
    hdr_exposures = 3
    hdr_ev_steps = 2.0

    def __init__(self, min_focus_score=50.0,
                       enable_hdr=False):
        self.min_focus_score = min_focus_score
        self.enable_hdr = enable_hdr

