"""Image pipeline — subject-aware multi-crop with P4 validation.

Flow:
  1. Make proxy (≤480p)
  2. Detect subjects on proxy (YuNet → Haar → saliency)
  3. Score subjects by importance
  4. Compute weighted focus point
  5. For each target ratio, compute subject-aware crop box
  6. Write crop + metadata JSON
  7. Run P4 validation (dimensions, face visibility, margins, consistency)
  8. Mark asset as ready only if validation passes
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import cv2
import numpy as np

from src.detection import (
    detect_subjects,
    DetectedSubject,
    score_subjects,
    weighted_focus,
)
from src.utils import ASPECT_RATIOS, PROXY_MAX_DIM, OUTPUTS_DIR
from src.utils.geometry import (
    compute_subject_aware_crop,
    scale_roi,
    faces_visible_in_crop,
)
from src.validation import validate_image_asset


def _make_proxy(image: np.ndarray, max_dim: int = PROXY_MAX_DIM):
    """Resize so longest edge ≤ max_dim. Returns (proxy, scale_factor)."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image.copy(), 1.0
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def process_image(
    image_path: str | Path,
    output_dir: str | Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Generate all aspect-ratio crops for a single image.

    Returns:
        {ratio_label: {path, crop_box, metadata, validation, ready}}
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    src_h, src_w = image.shape[:2]

    # --- Proxy detection -------------------------------------------------
    proxy, scale = _make_proxy(image)
    ph, pw = proxy.shape[:2]
    subjects_proxy = detect_subjects(proxy)

    # Map subject boxes back to full resolution
    subjects_full: list[DetectedSubject] = []
    for s in subjects_proxy:
        fx1, fy1, fx2, fy2 = scale_roi(
            (s.x1, s.y1, s.x2, s.y2), (pw, ph), (src_w, src_h)
        )
        subjects_full.append(DetectedSubject(
            x1=fx1, y1=fy1, x2=fx2, y2=fy2,
            confidence=s.confidence, category=s.category, source=s.source,
        ))

    scores = score_subjects(subjects_full, src_w, src_h)
    focus_x, focus_y = weighted_focus(subjects_full, scores, src_w, src_h)

    face_boxes = [(s.x1, s.y1, s.x2, s.y2) for s in subjects_full if s.category == "face"]

    # --- Generate crops ---------------------------------------------------
    results: Dict[str, Dict[str, Any]] = {}
    for label, (rw, rh) in ASPECT_RATIOS.items():
        crop_box = compute_subject_aware_crop(
            src_w, src_h, rw, rh, focus_x, focus_y,
            subject_boxes=face_boxes if face_boxes else None,
        )
        x1, y1, x2, y2 = crop_box
        crop = image[y1:y2, x1:x2]

        safe_label = label.replace(":", "x")
        out_path = output_dir / f"{image_path.stem}_{safe_label}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Face visibility for this crop
        vis = faces_visible_in_crop(crop_box, face_boxes) if face_boxes else []

        # Metadata
        meta = {
            "source": str(image_path),
            "source_size": {"w": src_w, "h": src_h},
            "target_ratio": label,
            "crop_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "crop_size": {"w": x2 - x1, "h": y2 - y1},
            "focus": {"x": round(focus_x, 4), "y": round(focus_y, 4)},
            "detected_subjects": [
                {**s.to_dict(), "importance": sc}
                for s, sc in zip(subjects_full, scores)
            ],
            "face_visibility": vis,
        }

        meta_path = output_dir / f"{image_path.stem}_{safe_label}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        # P4 Validation
        val = validate_image_asset(out_path, metadata=meta, expected_ratio=(rw, rh))

        # Asset is ready only if validation passes
        ready = val["status"] == "PASS"

        results[label] = {
            "path": out_path,
            "meta_path": meta_path,
            "crop_box": crop_box,
            "metadata": meta,
            "validation": val,
            "ready": ready,
        }

    return results
