"""Output validation — machine-readable checks on generated assets."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import cv2


def validate_image(
    path: str | Path,
    expected_ratio: tuple[int, int] | None = None,
    ratio_tolerance: float = 0.02,
) -> Dict:
    """Validate a generated image asset.

    Returns {ok: bool, checks: [{name, passed, detail}]}.
    """
    path = Path(path)
    checks: List[Dict] = []

    # File exists & readable
    exists = path.exists() and path.stat().st_size > 0
    checks.append({"name": "file_exists", "passed": exists, "detail": str(path)})
    if not exists:
        return {"ok": False, "checks": checks}

    img = cv2.imread(str(path))
    readable = img is not None
    checks.append({"name": "readable", "passed": readable})
    if not readable:
        return {"ok": False, "checks": checks}

    h, w = img.shape[:2]
    checks.append({
        "name": "min_resolution",
        "passed": w >= 100 and h >= 100,
        "detail": f"{w}x{h}",
    })

    if expected_ratio:
        actual = w / h
        expected = expected_ratio[0] / expected_ratio[1]
        diff = abs(actual - expected) / expected
        checks.append({
            "name": "aspect_ratio",
            "passed": diff <= ratio_tolerance,
            "detail": f"actual={actual:.3f} expected={expected:.3f} diff={diff:.4f}",
        })

    ok = all(c["passed"] for c in checks)
    return {"ok": ok, "checks": checks}


def validate_video(path: str | Path) -> Dict:
    """Basic validation of a generated video file."""
    path = Path(path)
    checks: List[Dict] = []

    exists = path.exists() and path.stat().st_size > 0
    checks.append({"name": "file_exists", "passed": exists, "detail": str(path)})
    if not exists:
        return {"ok": False, "checks": checks}

    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    checks.append({"name": "openable", "passed": opened})
    if opened:
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        checks.append({
            "name": "has_frames",
            "passed": fc > 0,
            "detail": f"{fc} frames",
        })
    cap.release()

    ok = all(c["passed"] for c in checks)
    return {"ok": ok, "checks": checks}
