import numpy as np
import torch
from torch.serialization import safe_globals

try:
    from numpy._core.multiarray import _reconstruct
except ImportError:  # NumPy < 2.0
    from numpy.core.multiarray import _reconstruct


def load_numpy_torch_artifact(path, map_location=None):
    """Load trusted NumPy-backed torch artifacts under PyTorch 2.6+ safe mode."""
    numpy_dtype_classes = [
        type(np.dtype("float32")),
        type(np.dtype("float64")),
        type(np.dtype("int32")),
        type(np.dtype("int64")),
    ]
    with safe_globals([_reconstruct, np.ndarray, np.dtype, *numpy_dtype_classes]):
        return torch.load(path, map_location=map_location, weights_only=True)


def load_state_dict_checkpoint(path, map_location=None):
    """Load trusted model state-dict checkpoints with PyTorch's restricted unpickler."""
    return torch.load(path, map_location=map_location, weights_only=True)
