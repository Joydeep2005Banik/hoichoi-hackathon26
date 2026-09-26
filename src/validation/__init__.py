"""P4 Asset Validation — comprehensive checks for generated crops.

Validates:
- Expected dimensions and aspect ratio
- Subject/face visibility within crop
- Crop boundary violations (subjects cut off)
- Safe margins around edges
- File integrity and readability
- Source/target metadata consistency
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import json


def validate_image_asset(
    path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
    expected_ratio: tuple[int, int] | None = None,
    ratio_tolerance: float = 0.02,
    min_face_visibility: float = 0.6,
    safe_margin_px: int = 20,
) -> Dict[str, Any]:
    """Comprehensive P4 validation for a generated image crop.

    Args:
        path: Path to generated crop
        metadata: Metadata dict from image pipeline (with crop_box, detected_subjects, etc.)
        expected_ratio: (w, h) target ratio
        ratio_tolerance: Allowed deviation from target ratio
        min_face_visibility: Minimum fraction of face area that must be visible
        safe_margin_px: Minimum distance from crop edge for important subjects

    Returns:
        {
            "status": "PASS" | "FAIL",
            "checks": [{name, passed, severity, detail}],
            "warnings": [{message, detail}],
            "metrics": {dimensions, face_visibility_score, etc.}
        }
    """
    path = Path(path)
    checks: List[Dict] = []
    warnings: List[Dict] = []
    metrics: Dict[str, Any] = {}

    # ── File integrity ────────────────────────────────────────
    exists = path.exists() and path.stat().st_size > 0
    checks.append({
        "name": "file_exists",
        "passed": exists,
        "severity": "critical",
        "detail": str(path) if not exists else None,
    })
    if not exists:
        return {"status": "FAIL", "checks": checks, "warnings": warnings, "metrics": metrics}

    img = cv2.imread(str(path))
    readable = img is not None
    checks.append({
        "name": "readable",
        "passed": readable,
        "severity": "critical",
    })
    if not readable:
        return {"status": "FAIL", "checks": checks, "warnings": warnings, "metrics": metrics}

    h, w = img.shape[:2]
    metrics["dimensions"] = {"w": w, "h": h}

    # ── Dimension checks ──────────────────────────────────────
    min_res_pass = w >= 100 and h >= 100
    checks.append({
        "name": "min_resolution",
        "passed": min_res_pass,
        "severity": "critical",
        "detail": f"{w}×{h}",
    })

    # ── Aspect ratio ──────────────────────────────────────────
    if expected_ratio:
        actual = w / h
        expected = expected_ratio[0] / expected_ratio[1]
        diff = abs(actual - expected) / expected
        ratio_pass = diff <= ratio_tolerance
        checks.append({
            "name": "aspect_ratio",
            "passed": ratio_pass,
            "severity": "critical",
            "detail": f"actual={actual:.4f} expected={expected:.4f} diff={diff*100:.2f}%",
        })
        metrics["aspect_ratio"] = {
            "actual": round(actual, 4),
            "expected": round(expected, 4),
            "deviation_pct": round(diff * 100, 2),
        }

    # ── File size sanity ──────────────────────────────────────
    size_kb = path.stat().st_size / 1024
    size_pass = size_kb >= 1
    checks.append({
        "name": "file_size",
        "passed": size_pass,
        "severity": "error",
        "detail": f"{size_kb:.1f} KB",
    })
    metrics["file_size_kb"] = round(size_kb, 1)

    # ── Metadata-driven checks ────────────────────────────────
    if metadata:
        # Source/target consistency
        crop_box = metadata.get("crop_box")
        source_size = metadata.get("source_size")
        if crop_box and source_size:
            crop_w = crop_box["x2"] - crop_box["x1"]
            crop_h = crop_box["y2"] - crop_box["y1"]
            consistency_pass = (crop_w == w and crop_h == h)
            checks.append({
                "name": "metadata_consistency",
                "passed": consistency_pass,
                "severity": "error",
                "detail": f"meta says {crop_w}×{crop_h}, actual {w}×{h}",
            })

            # Crop boundary violations (crop box must be within source bounds)
            src_w = source_size["w"]
            src_h = source_size["h"]
            boundary_pass = (
                crop_box["x1"] >= 0 and crop_box["y1"] >= 0 and
                crop_box["x2"] <= src_w and crop_box["y2"] <= src_h
            )
            checks.append({
                "name": "crop_boundary",
                "passed": boundary_pass,
                "severity": "critical",
                "detail": f"crop [{crop_box['x1']},{crop_box['y1']}:{crop_box['x2']},{crop_box['y2']}] vs source {src_w}×{src_h}",
            })

        # Face/subject visibility
        detected_subjects = metadata.get("detected_subjects", [])
        face_visibility = metadata.get("face_visibility", [])
        faces = [s for s in detected_subjects if s.get("category") == "face"]

        if faces and face_visibility:
            visible_count = sum(face_visibility)
            total_faces = len(face_visibility)
            visibility_ratio = visible_count / total_faces if total_faces > 0 else 1.0

            vis_pass = visibility_ratio >= min_face_visibility
            checks.append({
                "name": "face_visibility",
                "passed": vis_pass,
                "severity": "error",
                "detail": f"{visible_count}/{total_faces} faces visible (min {min_face_visibility*100:.0f}%)",
            })
            metrics["face_visibility_score"] = round(visibility_ratio, 2)

            if not vis_pass:
                warnings.append({
                    "message": "Some faces are cropped out",
                    "detail": f"Only {visible_count}/{total_faces} faces meet visibility threshold",
                })

        # Safe margins (check if important subjects are too close to edges)
        if crop_box and detected_subjects:
            subjects_near_edge = []
            for s in detected_subjects:
                if s.get("category") != "face":
                    continue
                if not all(k in s for k in ("x1", "y1", "x2", "y2")):
                    continue
                # Map subject box to crop-relative coords
                sx1 = s["x1"] - crop_box["x1"]
                sy1 = s["y1"] - crop_box["y1"]
                sx2 = s["x2"] - crop_box["x1"]
                sy2 = s["y2"] - crop_box["y1"]

                # Check if subject is too close to any edge
                near_left = sx1 < safe_margin_px
                near_top = sy1 < safe_margin_px
                near_right = (w - sx2) < safe_margin_px
                near_bottom = (h - sy2) < safe_margin_px

                if any([near_left, near_top, near_right, near_bottom]):
                    subjects_near_edge.append({
                        "subject": s.get("source", "unknown"),
                        "edges": [e for e, v in [("left", near_left), ("top", near_top),
                                                   ("right", near_right), ("bottom", near_bottom)] if v],
                    })

            safe_margin_pass = len(subjects_near_edge) == 0
            checks.append({
                "name": "safe_margins",
                "passed": safe_margin_pass,
                "severity": "warning",
                "detail": f"{len(subjects_near_edge)} subjects within {safe_margin_px}px of edge",
            })

            if subjects_near_edge:
                for subj in subjects_near_edge:
                    warnings.append({
                        "message": f"Subject near edge: {subj['subject']}",
                        "detail": f"Too close to {', '.join(subj['edges'])}",
                    })

    # ── Overall status ────────────────────────────────────────
    critical_failed = any(c["passed"] is False and c["severity"] == "critical" for c in checks)
    error_failed = any(c["passed"] is False and c["severity"] == "error" for c in checks)
    status = "FAIL" if (critical_failed or error_failed) else "PASS"

    return {
        "status": status,
        "ok": status == "PASS",
        "checks": checks,
        "warnings": warnings,
        "metrics": metrics,
    }


def validate_image(
    path: str | Path,
    expected_ratio: tuple[int, int] | None = None,
    ratio_tolerance: float = 0.02,
) -> Dict:
    """Legacy validation API — basic checks only.

    Returns {ok: bool, checks: [{name, passed, detail}]}.
    """
    path = Path(path)
    checks: List[Dict] = []

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

    size_kb = path.stat().st_size / 1024
    checks.append({
        "name": "file_size",
        "passed": size_kb >= 1,
        "detail": f"{size_kb:.1f} KB",
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
