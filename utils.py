from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch to make runs reproducible.
    Note: exact reproducibility can still vary across hardware/OS/library versions.
    """
    random.seed(seed)                      # Python RNG
    np.random.seed(seed)                   # NumPy RNG
    torch.manual_seed(seed)                # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)       # PyTorch GPU RNGs (all devices)

    # For CUDA backends: favor determinism over speed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------- Optional helpers (not required by main.py) --------

def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Return total parameter count. If trainable_only=True, only counts params with requires_grad.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't already exist.
    """
    os.makedirs(path, exist_ok=True)