"""Default experiment settings. These match the values used in the report."""

import torch

DEFAULTS = {
    "img_size": 224,
    "batch_size": 32,
    "num_epochs": 15,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "num_workers": 0,          # 0 keeps it compatible with macOS / MPS
    "min_votes": 5,            # keep places with more than this many votes
    "threshold_std": 1.5,      # unsafe < mean - k*std, safe > mean + k*std
    "val_fraction": 0.2,       # share of places held out for validation
    "seed": 42,
}

# name -> (architecture, colour mode)
EXPERIMENTS = {
    "resnet50_rgb": ("resnet50", "rgb"),
    "vit_rgb": ("vit_b16", "rgb"),
    "resnet50_gray": ("resnet50", "gray"),
    "vit_gray": ("vit_b16", "gray"),
}

DISPLAY_NAMES = {
    "resnet50_rgb": "RGB + ResNet50",
    "vit_rgb": "RGB + ViT-B/16",
    "resnet50_gray": "Grayscale + ResNet50",
    "vit_gray": "Grayscale + ViT-B/16",
}


def pick_device() -> str:
    """Prefer Apple Silicon GPU, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
