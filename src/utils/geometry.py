"""Geometry helpers for ROI mapping and crop computation."""
from __future__ import annotations
from typing import List, Tuple


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
    # Defensive clamp: ensure focus coordinates are within valid [0.0, 1.0] range
    focus_x = max(0.0, min(float(focus_x), 1.0))
    focus_y = max(0.0, min(float(focus_y), 1.0))

    target_ratio = ratio_w / ratio_h
    src_ratio = img_w / img_h

    if target_ratio >= src_ratio:
        crop_w = img_w
        crop_h = int(img_w / target_ratio)
    else:
        crop_h = img_h
        crop_w = int(img_h * target_ratio)

    cx = int(focus_x * img_w)
    cy = int(focus_y * img_h)

    x1 = clamp(cx - crop_w // 2, 0, img_w - crop_w)
    y1 = clamp(cy - crop_h // 2, 0, img_h - crop_h)
    return x1, y1, x1 + crop_w, y1 + crop_h


def compute_subject_aware_crop(
    img_w: int,
    img_h: int,
    ratio_w: int,
    ratio_h: int,
    focus_x: float,
    focus_y: float,
    subject_boxes: List[Tuple[int, int, int, int]] | None = None,
) -> Tuple[int, int, int, int]:
    """Like compute_crop_box but then nudges to include as many subject
    bounding boxes as possible.

    subject_boxes: list of (x1, y1, x2, y2) in pixel coords.
    """
    x1, y1, x2, y2 = compute_crop_box(img_w, img_h, ratio_w, ratio_h, focus_x, focus_y)
    crop_w = x2 - x1
    crop_h = y2 - y1

    if not subject_boxes:
        return x1, y1, x2, y2

    # Compute the bounding box of all subjects
    all_x1 = min(b[0] for b in subject_boxes)
    all_y1 = min(b[1] for b in subject_boxes)
    all_x2 = max(b[2] for b in subject_boxes)
    all_y2 = max(b[3] for b in subject_boxes)

    # If all subjects fit in the crop, just nudge to include them
    subj_w = all_x2 - all_x1
    subj_h = all_y2 - all_y1

    if subj_w <= crop_w and subj_h <= crop_h:
        # Ideal center of subjects
        scx = (all_x1 + all_x2) // 2
        scy = (all_y1 + all_y2) // 2

        # Blend: 70% subject center, 30% weighted focus
        bcx = int(0.7 * scx + 0.3 * focus_x * img_w)
        bcy = int(0.7 * scy + 0.3 * focus_y * img_h)

        nx1 = clamp(bcx - crop_w // 2, 0, img_w - crop_w)
        ny1 = clamp(bcy - crop_h // 2, 0, img_h - crop_h)

        # Verify all subjects still inside; if not, fall back to original
        if nx1 <= all_x1 and ny1 <= all_y1 and nx1 + crop_w >= all_x2 and ny1 + crop_h >= all_y2:
            return nx1, ny1, nx1 + crop_w, ny1 + crop_h

        # Try pure subject-center
        nx1 = clamp(scx - crop_w // 2, 0, img_w - crop_w)
        ny1 = clamp(scy - crop_h // 2, 0, img_h - crop_h)
        if nx1 <= all_x1 and ny1 <= all_y1 and nx1 + crop_w >= all_x2 and ny1 + crop_h >= all_y2:
            return nx1, ny1, nx1 + crop_w, ny1 + crop_h

    # Subjects too spread — use weighted focus (best effort)
    return x1, y1, x2, y2


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


def faces_visible_in_crop(
    crop_box: Tuple[int, int, int, int],
    face_boxes: List[Tuple[int, int, int, int]],
    min_overlap: float = 0.6,
) -> List[bool]:
    """For each face box, check whether at least min_overlap fraction of its
    area is inside the crop box."""
    cx1, cy1, cx2, cy2 = crop_box
    results = []
    for fx1, fy1, fx2, fy2 in face_boxes:
        face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
        # intersection
        ix1 = max(cx1, fx1)
        iy1 = max(cy1, fy1)
        ix2 = min(cx2, fx2)
        iy2 = min(cy2, fy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        results.append(inter / face_area >= min_overlap)
    return results
