"""Regression tests for crop path bounds safety.

Ensures invalid normalized focus coordinates cannot produce out-of-bounds crops.
"""
import pytest
from src.utils.geometry import compute_crop_box


class TestCropBoundsSafety:
    """Defensive bounds-checking for focus coordinates."""

    def test_valid_focus_unchanged(self):
        """Values inside [0,1] remain unchanged."""
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, 0.5, 0.5)
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080

        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, 0.0, 0.0)
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080

        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, 1.0, 1.0)
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080

    def test_out_of_bounds_focus_clamped(self):
        """Values outside [0,1] cannot produce an invalid crop."""
        # focus_x > 1.0
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, 1.5, 0.5)
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080

        # focus_x < 0.0
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, -0.3, 0.5)
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080

        # focus_y > 1.0
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, 0.5, 1.8)
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080

        # focus_y < 0.0
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, 0.5, -0.5)
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080

    def test_extreme_out_of_bounds(self):
        """Extreme out-of-bounds values are safely clamped."""
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, 999.0, -999.0)
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080

    def test_no_speaker_period_holds_valid_position(self):
        """Simulates render_reel behavior during no-speaker periods."""
        # Simulate render_reel's last_fx/last_fy holding logic
        last_fx, last_fy = 0.5, 0.5

        # Valid update
        raw_fx, raw_fy = 0.7, 0.4
        if 0.0 <= raw_fx <= 1.0 and 0.0 <= raw_fy <= 1.0:
            last_fx, last_fy = raw_fx, raw_fy
        assert last_fx == 0.7
        assert last_fy == 0.4

        # Invalid update — should hold previous valid position
        raw_fx, raw_fy = 1.5, -0.2
        if 0.0 <= raw_fx <= 1.0 and 0.0 <= raw_fy <= 1.0:
            last_fx, last_fy = raw_fx, raw_fy
        assert last_fx == 0.7  # Held
        assert last_fy == 0.4  # Held

        # Crop box with held position is valid
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, last_fx, last_fy)
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080

    def test_crop_dimensions_correct(self):
        """Crop box always produces correct 9:16 dimensions."""
        x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, 0.5, 0.5)
        crop_w = x2 - x1
        crop_h = y2 - y1
        # For 1920x1080 source, 9:16 crop should be 607x1080 (limited by height)
        assert crop_w == int(1080 * 9 / 16)
        assert crop_h == 1080

    def test_all_aspect_ratios_safe(self):
        """All standard aspect ratios produce valid crops with out-of-bounds focus."""
        ratios = [(16, 9), (1, 1), (4, 5), (9, 16)]
        for rw, rh in ratios:
            x1, y1, x2, y2 = compute_crop_box(1920, 1080, rw, rh, 1.5, -0.5)
            assert 0 <= x1 < x2 <= 1920, f"Failed for {rw}:{rh}"
            assert 0 <= y1 < y2 <= 1080, f"Failed for {rw}:{rh}"
