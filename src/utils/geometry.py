"""Geometry helpers for ROI mapping and crop computation."""
from __future__ import annotations
from typing import Tuple


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


def compute_crop_box(
    img_w: int,
    img_h: int,
    ratio_w: int,
    ratio_h: int,
    focus_x: float = 0.5,
    focus_y: float = 0.5,
) -> Tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for the largest crop of ratio_w:ratio_h
    centered on the focus point (0-1 normalized)."""
    target_ratio = ratio_w / ratio_h
    src_ratio = img_w / img_h

    if target_ratio >= src_ratio:
        # width-limited
        crop_w = img_w
        crop_h = int(img_w / target_ratio)
    else:
        # height-limited
        crop_h = img_h
        crop_w = int(img_h * target_ratio)

    cx = int(focus_x * img_w)
    cy = int(focus_y * img_h)

    x1 = clamp(cx - crop_w // 2, 0, img_w - crop_w)
    y1 = clamp(cy - crop_h // 2, 0, img_h - crop_h)
    return x1, y1, x1 + crop_w, y1 + crop_h


def scale_roi(
    roi: Tuple[int, int, int, int],
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Map an ROI from proxy resolution to full resolution."""
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    x1, y1, x2, y2 = roi
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
