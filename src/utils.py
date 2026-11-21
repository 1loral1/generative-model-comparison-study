"""Shared settings: device, output folders, seeding and history files."""
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.utils import make_grid, save_image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
HIST_DIR = OUT_DIR / "histories"
SAMPLE_DIR = OUT_DIR / "samples"
FIG_DIR = OUT_DIR / "figures"
METRIC_DIR = OUT_DIR / "metrics"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    """Seed every random number generator used in the project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_history(name, history):
    with open(HIST_DIR / f"{name}.json", "w") as f:
        json.dump(history, f)

def load_history(name):
    with open(HIST_DIR / f"{name}.json") as f:
        return json.load(f)



def show_grid(imgs, nrow, title=None, path=None):
    """imgs: (N,1,28,28) in [0,1]."""
    grid = make_grid(imgs.cpu(), nrow=nrow, padding=2, pad_value=1.0)
    plt.figure(figsize=(nrow, len(imgs) / nrow))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()