import torch
from monai.utils import set_determinism

"""Determine the device to be used for computation."""
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_runtime(seed: int):
    set_determinism(seed=seed)
    device = get_device()
    print(f"Using device: {device}")
    return device
