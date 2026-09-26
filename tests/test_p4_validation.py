"""Tests for P4 asset validation layer."""
import sys
from pathlib import Path
import json
import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestP4Validation:
    """Tests for the comprehensive P4 validation layer."""

    def test_validate_passing_asset(self, tmp_path):
        """Valid crop with good face visibility should pass."""
        from src.validation import validate_image_asset

        # Create a 1:1 crop (1080x1080)
        img = np.full((1080, 1080, 3), 128, dtype=np.uint8)
        path = tmp_path / "crop.jpg"
        cv2.imwrite(str(path), img)

        # Mock metadata with face well inside crop
        meta = {
            "source_size": {"w": 1920, "h": 1080},
            "crop_box": {"x1": 420, "y1": 0, "x2": 1500, "y2": 1080},
            "crop_size": {"w": 1080, "h": 1080},
            "detected_subjects": [
                {"x1": 500, "y1": 200, "x2": 700, "y2": 500, "category": "face", "source": "yunet"}
            ],
            "face_visibility": [True],
        }

        result = validate_image_asset(path, metadata=meta, expected_ratio=(1, 1))

        assert result["status"] == "PASS"
        assert all(c["passed"] for c in result["checks"])
        assert len(result["warnings"]) == 0

    def test_validate_missing_file(self):
        """Missing file should fail with critical severity."""
        from src.validation import validate_image_asset

        result = validate_image_asset("/nonexistent/file.jpg")

        assert result["status"] == "FAIL"
        assert result["checks"][0]["name"] == "file_exists"
        assert result["checks"][0]["passed"] is False
        assert result["checks"][0]["severity"] == "critical"

    def test_validate_bad_aspect_ratio(self, tmp_path):
        """Wrong aspect ratio should fail."""
        from src.validation import validate_image_asset

        # Create 1:1 image but claim it should be 16:9
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        path = tmp_path / "square.jpg"
        cv2.imwrite(str(path), img)

        result = validate_image_asset(path, expected_ratio=(16, 9))

        assert result["status"] == "FAIL"
        ratio_check = [c for c in result["checks"] if c["name"] == "aspect_ratio"][0]
        assert ratio_check["passed"] is False
        assert ratio_check["severity"] == "critical"

    def test_validate_face_visibility_fail(self, tmp_path):
        """Crop with poor face visibility should fail."""
        from src.validation import validate_image_asset

        img = np.zeros((1080, 607, 3), dtype=np.uint8)
        path = tmp_path / "crop.jpg"
        cv2.imwrite(str(path), img)

        # 2 faces detected but only 1 visible (50% < 60% threshold)
        meta = {
            "source_size": {"w": 1920, "h": 1080},
            "crop_box": {"x1": 657, "y1": 0, "x2": 1264, "y2": 1080},
            "detected_subjects": [
                {"x1": 200, "y1": 300, "x2": 400, "y2": 600, "category": "face"},
                {"x1": 1500, "y1": 300, "x2": 1700, "y2": 600, "category": "face"},
            ],
            "face_visibility": [True, False],  # Only 50% visible
        }

        result = validate_image_asset(path, metadata=meta, expected_ratio=(9, 16))

        assert result["status"] == "FAIL"
        vis_check = [c for c in result["checks"] if c["name"] == "face_visibility"][0]
        assert vis_check["passed"] is False
        assert vis_check["severity"] == "error"
        assert result["metrics"]["face_visibility_score"] == 0.5

    def test_validate_crop_boundary_violation(self, tmp_path):
        """Crop box exceeding source bounds should fail."""
        from src.validation import validate_image_asset

        img = np.zeros((1080, 1080, 3), dtype=np.uint8)
        path = tmp_path / "crop.jpg"
        cv2.imwrite(str(path), img)

        # Crop box goes beyond source image bounds
        meta = {
            "source_size": {"w": 1920, "h": 1080},
            "crop_box": {"x1": 1000, "y1": 0, "x2": 2100, "y2": 1080},  # x2 > 1920
            "crop_size": {"w": 1100, "h": 1080},
        }

        result = validate_image_asset(path, metadata=meta)

        assert result["status"] == "FAIL"
        boundary_check = [c for c in result["checks"] if c["name"] == "crop_boundary"][0]
        assert boundary_check["passed"] is False
        assert boundary_check["severity"] == "critical"

    def test_validate_metadata_consistency(self, tmp_path):
        """Mismatch between metadata and actual dimensions should fail."""
        from src.validation import validate_image_asset

        img = np.zeros((1080, 1080, 3), dtype=np.uint8)
        path = tmp_path / "crop.jpg"
        cv2.imwrite(str(path), img)

        # Metadata claims different size
        meta = {
            "source_size": {"w": 1920, "h": 1080},
            "crop_box": {"x1": 420, "y1": 0, "x2": 1320, "y2": 900},  # 900x900
            "crop_size": {"w": 900, "h": 900},
        }

        result = validate_image_asset(path, metadata=meta)

        assert result["status"] == "FAIL"
        consistency_check = [c for c in result["checks"] if c["name"] == "metadata_consistency"][0]
        assert consistency_check["passed"] is False

    def test_validate_safe_margins_warning(self, tmp_path):
        """Face too close to edge should generate warning."""
        from src.validation import validate_image_asset

        img = np.zeros((1080, 607, 3), dtype=np.uint8)
        path = tmp_path / "crop.jpg"
        cv2.imwrite(str(path), img)

        # Face at edge of crop (in crop coords: x=5 is within 20px margin)
        meta = {
            "source_size": {"w": 1920, "h": 1080},
            "crop_box": {"x1": 657, "y1": 0, "x2": 1264, "y2": 1080},
            "detected_subjects": [
                {"x1": 662, "y1": 300, "x2": 862, "y2": 600, "category": "face", "source": "yunet"}
            ],
            "face_visibility": [True],
        }

        result = validate_image_asset(path, metadata=meta, expected_ratio=(9, 16))

        # Should pass but have warnings
        assert result["status"] == "PASS"
        margin_check = [c for c in result["checks"] if c["name"] == "safe_margins"][0]
        assert margin_check["passed"] is False
        assert margin_check["severity"] == "warning"
        assert len(result["warnings"]) > 0

    def test_validate_metrics_populated(self, tmp_path):
        """Validation should populate metrics dict."""
        from src.validation import validate_image_asset

        img = np.zeros((1080, 864, 3), dtype=np.uint8)  # h, w -> 1080x864 for 4:5
        path = tmp_path / "crop.jpg"
        cv2.imwrite(str(path), img)

        meta = {
            "source_size": {"w": 1920, "h": 1080},
            "crop_box": {"x1": 170, "y1": 0, "x2": 1034, "y2": 1080},
            "detected_subjects": [{"category": "face"}],
            "face_visibility": [True],
        }

        result = validate_image_asset(path, metadata=meta, expected_ratio=(4, 5))

        assert "metrics" in result
        assert "dimensions" in result["metrics"]
        assert result["metrics"]["dimensions"]["w"] == 864
        assert result["metrics"]["dimensions"]["h"] == 1080
        assert "aspect_ratio" in result["metrics"]
        assert "file_size_kb" in result["metrics"]
        assert "face_visibility_score" in result["metrics"]

    def test_validate_no_metadata_still_works(self, tmp_path):
        """Validation without metadata should check basic properties only."""
        from src.validation import validate_image_asset

        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        path = tmp_path / "img.jpg"
        cv2.imwrite(str(path), img)

        result = validate_image_asset(path, expected_ratio=(16, 9))

        assert result["status"] == "PASS"
        # Should have basic checks but not metadata-specific ones
        check_names = {c["name"] for c in result["checks"]}
        assert "file_exists" in check_names
        assert "readable" in check_names
        assert "aspect_ratio" in check_names
        assert "face_visibility" not in check_names
        assert "crop_boundary" not in check_names

    def test_validate_small_file_fails(self, tmp_path):
        """Suspiciously small file should fail."""
        from src.validation import validate_image_asset

        # Create a tiny 10x10 image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        path = tmp_path / "tiny.jpg"
        cv2.imwrite(str(path), img)

        result = validate_image_asset(path)

        # Should fail on min_resolution
        res_check = [c for c in result["checks"] if c["name"] == "min_resolution"][0]
        assert res_check["passed"] is False


class TestPipelineIntegration:
    """Test P4 validation integrated into image pipeline."""

    def test_pipeline_marks_ready_assets(self, tmp_path):
        """Pipeline should mark assets as ready only after validation passes."""
        from src.image_pipeline import process_image

        # Create test image with a face pattern
        img = np.full((1080, 1920, 3), 80, dtype=np.uint8)
        cv2.circle(img, (600, 400), 100, (200, 180, 160), -1)
        cv2.circle(img, (570, 375), 12, (50, 50, 50), -1)
        cv2.circle(img, (630, 375), 12, (50, 50, 50), -1)

        src = tmp_path / "test.jpg"
        cv2.imwrite(str(src), img)

        results = process_image(src, output_dir=tmp_path / "out")

        for label, data in results.items():
            assert "ready" in data
            assert "validation" in data
            assert data["validation"]["status"] in ["PASS", "FAIL"]
            # ready should match validation status
            if data["validation"]["status"] == "PASS":
                assert data["ready"] is True
            else:
                assert data["ready"] is False

    def test_pipeline_validation_structure(self, tmp_path):
        """Pipeline validation output should have P4 structure."""
        from src.image_pipeline import process_image

        img = np.full((1080, 1920, 3), 100, dtype=np.uint8)
        src = tmp_path / "src.jpg"
        cv2.imwrite(str(src), img)

        results = process_image(src, output_dir=tmp_path / "out")
        val = results["16:9"]["validation"]

        # Check P4 structure
        assert "status" in val
        assert "checks" in val
        assert "warnings" in val
        assert "metrics" in val

        # Each check should have required fields
        for check in val["checks"]:
            assert "name" in check
            assert "passed" in check
            assert "severity" in check


class TestLegacyValidationAPI:
    """Ensure legacy validate_image API still works."""

    def test_legacy_validate_image_works(self, tmp_path):
        """Legacy API should still return {ok, checks}."""
        from src.validation import validate_image

        img = np.zeros((450, 800, 3), dtype=np.uint8)
        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), img)

        result = validate_image(path, expected_ratio=(16, 9))

        assert "ok" in result
        assert "checks" in result
        assert isinstance(result["ok"], bool)
        assert isinstance(result["checks"], list)
