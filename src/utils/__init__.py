"""Shared configuration constants."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ROOT_DIR / "models"

OUTPUTS_DIR.mkdir(exist_ok=True)

# Target aspect ratios as (w, h) tuples
ASPECT_RATIOS = {
    "16:9": (16, 9),
    "1:1": (1, 1),
    "4:5": (4, 5),
    "9:16": (9, 16),
}

# Proxy analysis resolution (longest edge)
PROXY_MAX_DIM = 480

# Video reel target
REEL_WIDTH = 1080
REEL_HEIGHT = 1920
REEL_FPS = 25
