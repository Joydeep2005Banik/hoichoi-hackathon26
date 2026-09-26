"""Tests for the image MVP — geometry, detection, pipeline, validation."""
import sys
from pathlib import Path
import json
import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Geometry tests ─────────────────────────────────────────

class TestCropBox:
    def test_center_square(self):
        from src.utils.geometry import compute_crop_box
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 1, 1, 0.5, 0.5)
        assert x2 - x1 == y2 - y1 == 1080

    def test_16x9_full_frame(self):
        from src.utils.geometry import compute_crop_box
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 16, 9, 0.5, 0.5)
        assert x2 - x1 == 1920
        assert y2 - y1 == 1080

    def test_9x16_tall(self):
        from src.utils.geometry import compute_crop_box
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, 0.5, 0.5)
        ratio = (x2 - x1) / (y2 - y1)
        assert abs(ratio - 9 / 16) < 0.01

    def test_focus_left_clamp(self):
        from src.utils.geometry import compute_crop_box
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 1, 1, 0.1, 0.5)
        assert x1 == 0

    def test_focus_right_clamp(self):
        from src.utils.geometry import compute_crop_box
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 1, 1, 0.95, 0.5)
        assert x2 == 1920

    def test_crop_never_exceeds_image(self):
        from src.utils.geometry import compute_crop_box
        for rw, rh in [(16, 9), (1, 1), (4, 5), (9, 16)]:
            for fx, fy in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)]:
                x1, y1, x2, y2 = compute_crop_box(1920, 1080, rw, rh, fx, fy)
                assert 0 <= x1 < x2 <= 1920
                assert 0 <= y1 < y2 <= 1080

    def test_4x5_ratio(self):
        from src.utils.geometry import compute_crop_box
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 4, 5, 0.5, 0.5)
        ratio = (x2 - x1) / (y2 - y1)
        assert abs(ratio - 4 / 5) < 0.01


class TestSubjectAwareCrop:
    def test_single_face_shifts_crop(self):
        from src.utils.geometry import compute_subject_aware_crop
        # Face at right side of 1920x1080
        face = (1500, 300, 1700, 600)
        x1, y1, x2, y2 = compute_subject_aware_crop(
            1920, 1080, 1, 1, 0.8, 0.4, [face]
        )
        # Face should be inside the crop
        assert x1 <= 1500 and x2 >= 1700

    def test_no_subjects_same_as_basic(self):
        from src.utils.geometry import compute_crop_box, compute_subject_aware_crop
        basic = compute_crop_box(1920, 1080, 1, 1, 0.5, 0.5)
        aware = compute_subject_aware_crop(1920, 1080, 1, 1, 0.5, 0.5, None)
        assert basic == aware

    def test_two_faces_both_included(self):
        from src.utils.geometry import compute_subject_aware_crop
        faces = [(200, 300, 400, 600), (600, 300, 800, 600)]
        x1, y1, x2, y2 = compute_subject_aware_crop(
            1920, 1080, 16, 9, 0.5, 0.5, faces
        )
        # Both faces should be inside
        for fx1, fy1, fx2, fy2 in faces:
            assert x1 <= fx1 and x2 >= fx2
            assert y1 <= fy1 and y2 >= fy2


class TestFaceVisibility:
    def test_face_fully_inside(self):
        from src.utils.geometry import faces_visible_in_crop
        assert faces_visible_in_crop((0, 0, 1000, 1000), [(100, 100, 300, 300)]) == [True]

    def test_face_fully_outside(self):
        from src.utils.geometry import faces_visible_in_crop
        assert faces_visible_in_crop((0, 0, 100, 100), [(500, 500, 700, 700)]) == [False]

    def test_face_partially_inside(self):
        from src.utils.geometry import faces_visible_in_crop
        # Face 200x200, crop covers half -> 50% overlap < 60% threshold
        result = faces_visible_in_crop((0, 0, 200, 200), [(100, 0, 300, 200)])
        assert result == [False]


class TestScaleROI:
    def test_basic_scale(self):
        from src.utils.geometry import scale_roi
        scaled = scale_roi((10, 20, 100, 200), (480, 270), (1920, 1080))
        assert scaled == (40, 80, 400, 800)


# ── Detection tests ────────────────────────────────────────

class TestDetection:
    def test_score_subjects_empty(self):
        from src.detection import score_subjects
        assert score_subjects([], 1920, 1080) == []

    def test_score_subjects_face_higher_than_saliency(self):
        from src.detection import DetectedSubject, score_subjects
        face = DetectedSubject(400, 200, 600, 500, 0.9, "face", "yunet")
        sal = DetectedSubject(400, 200, 600, 500, 0.9, "saliency", "saliency")
        scores = score_subjects([face, sal], 1920, 1080)
        assert scores[0] > scores[1]

    def test_weighted_focus_center(self):
        from src.detection import DetectedSubject, weighted_focus
        s = DetectedSubject(460, 240, 500, 300, 0.9, "face", "yunet")
        fx, fy = weighted_focus([s], [1.0], 960, 540)
        assert 0.0 <= fx <= 1.0
        assert 0.0 <= fy <= 1.0

    def test_weighted_focus_no_subjects(self):
        from src.detection import weighted_focus
        assert weighted_focus([], [], 1920, 1080) == (0.5, 0.5)

    def test_detect_subjects_returns_list(self):
        from src.detection import detect_subjects
        # Create a synthetic image with a face-like pattern
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 200), 80, (200, 180, 160), -1)  # head
        cv2.circle(img, (300, 185), 10, (50, 50, 50), -1)      # eyes
        cv2.circle(img, (340, 185), 10, (50, 50, 50), -1)
        subjects = detect_subjects(img)
        # Should return a list (may or may not detect face on synthetic)
        assert isinstance(subjects, list)
        # Saliency fallback should at least produce something
        assert len(subjects) >= 1

    def test_detected_subject_area(self):
        from src.detection import DetectedSubject
        s = DetectedSubject(10, 20, 110, 220, 0.9, "face", "yunet")
        assert s.area == 100 * 200

    def test_legacy_faces_center(self):
        from src.detection import faces_center
        assert faces_center([], 1920, 1080) == (0.5, 0.5)


# ── Validation tests ───────────────────────────────────────

class TestValidation:
    def test_missing_file(self):
        from src.validation import validate_image
        r = validate_image("/nonexistent/image.jpg")
        assert r["ok"] is False
        assert r["checks"][0]["name"] == "file_exists"

    def test_valid_image(self, tmp_path):
        from src.validation import validate_image
        img = np.zeros((450, 800, 3), dtype=np.uint8)  # 800/450 ≈ 1.778 ≈ 16:9
        p = tmp_path / "test.jpg"
        cv2.imwrite(str(p), img)
        r = validate_image(p, expected_ratio=(16, 9))
        assert r["ok"] is True
        assert any(c["name"] == "aspect_ratio" for c in r["checks"])

    def test_bad_ratio(self, tmp_path):
        from src.validation import validate_image
        img = np.zeros((500, 500, 3), dtype=np.uint8)  # 1:1
        p = tmp_path / "square.jpg"
        cv2.imwrite(str(p), img)
        r = validate_image(p, expected_ratio=(16, 9))
        ratio_check = [c for c in r["checks"] if c["name"] == "aspect_ratio"][0]
        assert ratio_check["passed"] is False

    def test_video_missing(self):
        from src.validation import validate_video
        r = validate_video("/nonexistent/video.mp4")
        assert r["ok"] is False


# ── Pipeline integration test ──────────────────────────────

class TestImagePipeline:
    def test_process_synthetic_image(self, tmp_path):
        """Full pipeline on a synthetic image with a face-like circle."""
        from src.image_pipeline import process_image

        # Create a 1920x1080 test image with a bright circle (subject)
        img = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        cv2.circle(img, (600, 400), 120, (200, 180, 160), -1)
        cv2.circle(img, (570, 375), 15, (50, 50, 50), -1)
        cv2.circle(img, (630, 375), 15, (50, 50, 50), -1)
        cv2.ellipse(img, (600, 430), (30, 15), 0, 0, 180, (50, 50, 50), 2)

        src_path = tmp_path / "test_master.jpg"
        cv2.imwrite(str(src_path), img)

        out_dir = tmp_path / "crops"
        results = process_image(src_path, output_dir=out_dir)

        assert len(results) == 4
        for label in ["16:9", "1:1", "4:5", "9:16"]:
            assert label in results
            data = results[label]
            # File was written
            assert data["path"].exists()
            # Metadata JSON was written
            assert data["meta_path"].exists()
            meta = json.loads(data["meta_path"].read_text())
            assert meta["target_ratio"] == label
            assert "detected_subjects" in meta
            assert "crop_box" in meta
            # Validation passed
            assert data["validation"]["ok"] is True

    def test_process_generates_correct_ratios(self, tmp_path):
        from src.image_pipeline import process_image

        img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        src = tmp_path / "ratio_test.jpg"
        cv2.imwrite(str(src), img)

        results = process_image(src, output_dir=tmp_path / "out")
        for label, (rw, rh) in [("16:9", (16, 9)), ("1:1", (1, 1)),
                                  ("4:5", (4, 5)), ("9:16", (9, 16))]:
            crop = cv2.imread(str(results[label]["path"]))
            h, w = crop.shape[:2]
            actual = w / h
            expected = rw / rh
            assert abs(actual - expected) / expected < 0.02, \
                f"{label}: actual={actual:.3f} expected={expected:.3f}"
