"""Face / subject detection — MediaPipe with Haar cascade fallback."""
from __future__ import annotations
from typing import List, Tuple, Optional
import cv2
import numpy as np

# Lazy-loaded MediaPipe detector
_mp_face_detection = None


def _get_mp_detector():
    global _mp_face_detection
    if _mp_face_detection is None:
        import mediapipe as mp
        _mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
    return _mp_face_detection


def detect_faces_mediapipe(
    image_rgb: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """Return list of (x1, y1, x2, y2) face boxes using MediaPipe."""
    detector = _get_mp_detector()
    results = detector.process(image_rgb)
    boxes = []
    if results.detections:
        h, w = image_rgb.shape[:2]
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, int((bb.xmin + bb.width) * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            boxes.append((x1, y1, x2, y2))
    return boxes


def detect_faces_haar(
    image_gray: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """Haar cascade fallback."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    rects = cascade.detectMultiScale(image_gray, 1.3, 5)
    return [(x, y, x + w, y + h) for (x, y, w, h) in rects]


def detect_faces(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces: try MediaPipe first, fall back to Haar."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes = detect_faces_mediapipe(rgb)
    if not boxes:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        boxes = detect_faces_haar(gray)
    return boxes


def faces_center(
    boxes: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> Tuple[float, float]:
    """Return normalized (0-1) center of mass of all face boxes.
    Falls back to image center if no boxes."""
    if not boxes:
        return 0.5, 0.5
    cx = sum((b[0] + b[2]) / 2 for b in boxes) / len(boxes)
    cy = sum((b[1] + b[3]) / 2 for b in boxes) / len(boxes)
    return cx / img_w, cy / img_h
