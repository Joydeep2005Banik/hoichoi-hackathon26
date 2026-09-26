"""Image pipeline — produce subject-aware crops for all target ratios."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import cv2

from src.detection import detect_faces, faces_center
from src.utils import ASPECT_RATIOS, PROXY_MAX_DIM, OUTPUTS_DIR
from src.utils.geometry import compute_crop_box, scale_roi


def _make_proxy(image: "np.ndarray", max_dim: int = PROXY_MAX_DIM):
    """Resize image so longest edge ≤ max_dim; return (proxy, scale_factor)."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image, 1.0
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def process_image(
    image_path: str | Path,
    output_dir: str | Path | None = None,
) -> Dict[str, Path]:
    """Generate all aspect-ratio crops for a single image.

    Returns dict mapping ratio label to output file path.
    """
    import numpy as np

    image_path = Path(image_path)
    output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]
    proxy, _ = _make_proxy(image)
    ph, pw = proxy.shape[:2]

    # Detect on proxy
    boxes_proxy = detect_faces(proxy)
    focus_x, focus_y = faces_center(boxes_proxy, pw, ph)

    outputs: Dict[str, Path] = {}
    for label, (rw, rh) in ASPECT_RATIOS.items():
        x1, y1, x2, y2 = compute_crop_box(w, h, rw, rh, focus_x, focus_y)
        crop = image[y1:y2, x1:x2]
        safe_label = label.replace(":", "x")
        out_path = output_dir / f"{image_path.stem}_{safe_label}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        outputs[label] = out_path

    return outputs
