from dataclasses import dataclass
from typing import Optional
from pecnet.preprocessing.Normalizers import *

@dataclass
class PreprocessArtifacts:
    # transformators on target
    target_scaler: Optional[Scaler] = None
    target_normalizer: Optional[Normalizer] = None

    # hyperparameters
    wavelet_type: str = "haar"
    sequence_size: int = 8
    error_sequence_size: int = 24
    stride: int = 1
    required_timestamps: Optional[int] = None
    test_ratio: float = 0.1
    conjoincy: bool = False
    target_normalization_type: str = "window_mean"