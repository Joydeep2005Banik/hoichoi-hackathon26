"""Face / subject detection with importance scoring.

Detection cascade:
  1. OpenCV YuNet (lightweight ONNX, ~230KB) — primary
  2. Haar cascade — fallback
  3. Saliency center — last resort

Each detector returns DetectedSubject entries with bounding box, confidence,
and a category tag used for importance weighting.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path

from src.utils import MODELS_DIR

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DetectedSubject:
    """A detected region of interest."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    category: str  # "face", "saliency"
    source: str    # "yunet", "haar", "saliency"

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["area"] = self.area
        return d

# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

# Category weights — faces are most important
_CATEGORY_WEIGHT = {"face": 1.0, "saliency": 0.4}


def score_subjects(
    subjects: List[DetectedSubject],
    img_w: int,
    img_h: int,
) -> List[float]:
    """Return an importance score for each subject.

    Score combines:
      - category weight (faces >> saliency)
      - confidence
      - relative area (larger face = more important)
      - centrality bonus (subjects near image center get a small boost)
    """
    if not subjects:
        return []
    img_area = img_w * img_h
    scores = []
    for s in subjects:
        cat_w = _CATEGORY_WEIGHT.get(s.category, 0.3)
        area_ratio = min(s.area / img_area, 0.5)  # cap at 50% of image
        # centrality: 1.0 at center, 0.7 at edges
        dx = abs(s.cx / img_w - 0.5)
        dy = abs(s.cy / img_h - 0.5)
        centrality = 1.0 - 0.3 * (dx + dy)
        score = cat_w * s.confidence * (0.5 + area_ratio) * centrality
        scores.append(round(score, 4))
    return scores


def weighted_focus(
    subjects: List[DetectedSubject],
    scores: List[float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float]:
    """Weighted center of mass (normalized 0-1) based on importance scores."""
    if not subjects or not scores:
        return 0.5, 0.5
    total = sum(scores)
    if total == 0:
        return 0.5, 0.5
    fx = sum(s.cx * w for s, w in zip(subjects, scores)) / total / img_w
    fy = sum(s.cy * w for s, w in zip(subjects, scores)) / total / img_h
    return fx, fy


# ---------------------------------------------------------------------------
# YuNet detector (primary)
# ---------------------------------------------------------------------------

_yunet_model_path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_yunet_detector = None


def _get_yunet(input_size: Tuple[int, int] = (320, 320)):
    """Lazy-load YuNet. Re-creates if input size changed."""
    global _yunet_detector
    if _yunet_detector is not None:
        # Check if size matches
        cur = _yunet_detector.getInputSize()
        if tuple(cur) == tuple(input_size):
            return _yunet_detector
    if not _yunet_model_path.exists():
        return None
    _yunet_detector = cv2.FaceDetectorYN.create(
        str(_yunet_model_path),
        "",
        input_size,
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=20,
    )
    return _yunet_detector


def detect_faces_yunet(image_bgr: np.ndarray) -> List[DetectedSubject]:
    """Detect faces using OpenCV YuNet (ONNX, ~230KB, CPU-fast)."""
    h, w = image_bgr.shape[:2]
    detector = _get_yunet((w, h))
    if detector is None:
        return []
    detector.setInputSize((w, h))
    _, faces = detector.detect(image_bgr)
    subjects = []
    if faces is not None:
        for face in faces:
            x1 = max(0, int(face[0]))
            y1 = max(0, int(face[1]))
            x2 = min(w, int(face[0] + face[2]))
            y2 = min(h, int(face[1] + face[3]))
            conf = float(face[14]) if face.shape[0] > 14 else float(face[-1])
            subjects.append(DetectedSubject(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=round(conf, 3),
                category="face", source="yunet",
            ))
    return subjects


# ---------------------------------------------------------------------------
# Haar cascade (fallback)
# ---------------------------------------------------------------------------

def detect_faces_haar(image_bgr: np.ndarray) -> List[DetectedSubject]:
    """Haar cascade fallback — lower quality but always available."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    rects = cascade.detectMultiScale(gray, 1.3, 5)
    return [
        DetectedSubject(
            x1=int(x), y1=int(y), x2=int(x + w), y2=int(y + h),
            confidence=0.6, category="face", source="haar",
        )
        for (x, y, w, h) in rects
    ]


# ---------------------------------------------------------------------------
# Saliency fallback
# ---------------------------------------------------------------------------

def detect_saliency_center(image_bgr: np.ndarray) -> List[DetectedSubject]:
    """Simple gradient-magnitude saliency as last-resort focus point."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)

    # Threshold top 20% of gradient energy
    thresh = np.percentile(mag, 80)
    mask = (mag >= thresh).astype(np.uint8)
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        h, w = image_bgr.shape[:2]
        return [DetectedSubject(
            x1=w // 4, y1=h // 4, x2=3 * w // 4, y2=3 * h // 4,
            confidence=0.2, category="saliency", source="saliency",
        )]

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    # Tighten to center 60% of salient blob
    cy, cx = coords.mean(axis=0)
    h_half = (y_max - y_min) * 0.3
    w_half = (x_max - x_min) * 0.3
    ih, iw = image_bgr.shape[:2]
    return [DetectedSubject(
        x1=max(0, int(cx - w_half)),
        y1=max(0, int(cy - h_half)),
        x2=min(iw, int(cx + w_half)),
        y2=min(ih, int(cy + h_half)),
        confidence=0.3, category="saliency", source="saliency",
    )]


# ---------------------------------------------------------------------------
# Unified cascade
# ---------------------------------------------------------------------------

def detect_subjects(image_bgr: np.ndarray) -> List[DetectedSubject]:
    """Run the full detection cascade: YuNet → Haar → saliency."""
    subjects = detect_faces_yunet(image_bgr)
    if subjects:
        return subjects
    subjects = detect_faces_haar(image_bgr)
    if subjects:
        return subjects
    return detect_saliency_center(image_bgr)


# ---------------------------------------------------------------------------
# Legacy helpers (kept for video_pipeline compatibility)
# ---------------------------------------------------------------------------

def detect_faces(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return face boxes as (x1, y1, x2, y2) tuples — legacy API."""
    subjects = detect_subjects(image_bgr)
    return [(s.x1, s.y1, s.x2, s.y2) for s in subjects if s.category == "face"]


def faces_center(
    boxes: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> Tuple[float, float]:
    """Normalized center of mass of face boxes. Falls back to (0.5, 0.5)."""
    if not boxes:
        return 0.5, 0.5
    cx = sum((b[0] + b[2]) / 2 for b in boxes) / len(boxes)
    cy = sum((b[1] + b[3]) / 2 for b in boxes) / len(boxes)
    return cx / img_w, cy / img_h
