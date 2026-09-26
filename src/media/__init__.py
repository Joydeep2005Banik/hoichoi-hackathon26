"""Media I/O helpers."""
from __future__ import annotations
from pathlib import Path
import cv2


def read_image(path: str | Path):
    """Read an image via OpenCV, raise on failure."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img
