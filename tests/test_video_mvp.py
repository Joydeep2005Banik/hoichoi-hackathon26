"""Tests for video MVP — tracking, analysis, and rendering."""
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestTracking:
    """Test the IoU-based tracker."""

    def test_iou_perfect_overlap(self):
        from src.tracking import compute_iou
        box = (100, 100, 200, 200)
        assert compute_iou(box, box) == 1.0

    def test_iou_no_overlap(self):
        from src.tracking import compute_iou
        assert compute_iou((0, 0, 100, 100), (200, 200, 300, 300)) == 0.0

    def test_iou_partial_overlap(self):
        from src.tracking import compute_iou
        # 50% overlap
        iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.14 < iou < 0.15

    def test_track_creation(self):
        from src.tracking import Track
        track = Track(0, (100, 100, 200, 200), 0)
        assert track.id == 0
        assert len(track.boxes) == 1
        assert track.current_box == (100, 100, 200, 200)

    def test_track_update(self):
        from src.tracking import Track
        track = Track(0, (100, 100, 200, 200), 0)
        track.update((110, 110, 210, 210), 10)
        assert len(track.boxes) == 2
        assert track.last_seen == 10
        assert track.missed_frames == 0

    def test_track_missed(self):
        from src.tracking import Track
        track = Track(0, (100, 100, 200, 200), 0)
        track.mark_missed(10)
        assert track.missed_frames == 1
        assert len(track.boxes) == 2  # Extrapolated position added

    def test_tracker_first_frame(self):
        from src.tracking import SimpleTracker
        tracker = SimpleTracker()
        detections = [(100, 100, 200, 200), (300, 100, 400, 200)]
        tracks = tracker.update(detections, 0)
        assert len(tracks) == 2

    def test_tracker_matching(self):
        from src.tracking import SimpleTracker
        tracker = SimpleTracker()

        # Frame 0: 2 detections
        det0 = [(100, 100, 200, 200), (300, 100, 400, 200)]
        tracker.update(det0, 0)

        # Frame 10: same subjects moved slightly
        det10 = [(110, 110, 210, 210), (310, 110, 410, 210)]
        tracks = tracker.update(det10, 10)

        # Should still be 2 tracks (not 4)
        assert len(tracks) == 2

    def test_tracker_stale_removal(self):
        from src.tracking import SimpleTracker
        tracker = SimpleTracker(max_age=3)

        # Frame 0
        tracker.update([(100, 100, 200, 200)], 0)
        assert len(tracker.tracks) == 1

        # Miss for max_age frames
        for i in range(1, 5):
            tracker.update([], i)

        # Track should be removed
        assert len(tracker.tracks) == 0

    def test_get_primary_track(self):
        from src.tracking import SimpleTracker
        tracker = SimpleTracker()

        # Create tracks with different lengths
        tracker.update([(100, 100, 200, 200), (300, 100, 400, 200)], 0)
        tracker.update([(105, 105, 205, 205)], 10)  # Only first track continues
        tracker.update([(110, 110, 210, 210)], 20)

        primary = tracker.get_primary_track()
        assert primary is not None
        assert len(primary.boxes) >= 3  # Longest track


class TestVideoPipeline:
    """Test video analysis and rendering."""

    def test_synthetic_video_analysis(self, tmp_path):
        """Test full pipeline on a synthetic video with moving face."""
        from src.video_pipeline import analyse_video, render_reel, extract_still

        # Create a 2-second synthetic video (50 frames @ 25 FPS)
        video_path = tmp_path / "test.mp4"
        fps = 25.0
        n_frames = 50
        w, h = 640, 480

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

        # Draw a moving circle (simulated face)
        for i in range(n_frames):
            frame = np.full((h, w, 3), 80, dtype=np.uint8)
            # Move from left to right
            cx = int(100 + i * 400 / n_frames)
            cy = 240
            cv2.circle(frame, (cx, cy), 60, (200, 180, 160), -1)
            cv2.circle(frame, (cx - 20, cy - 15), 8, (50, 50, 50), -1)
            cv2.circle(frame, (cx + 20, cy - 15), 8, (50, 50, 50), -1)
            out.write(frame)

        out.release()

        # Analyze
        cache_path = tmp_path / "analysis.json"
        analysis = analyse_video(video_path, sample_fps=5.0, cache_path=cache_path)

        # Check structure
        assert "metadata" in analysis
        assert "crop_path" in analysis
        assert analysis["metadata"]["fps"] == fps
        assert analysis["metadata"]["total_frames"] == n_frames

        # Should have detected something
        assert len(analysis["crop_path"]) > 0

        # Render reel
        reel_path = render_reel(video_path, analysis, output_path=tmp_path / "reel.mp4")
        assert reel_path.exists()

        # Verify reel dimensions
        cap = cv2.VideoCapture(str(reel_path))
        reel_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        reel_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        assert reel_w == 1080
        assert reel_h == 1920

        # Extract still
        still_path = extract_still(video_path, analysis, output_path=tmp_path / "still.jpg")
        assert still_path.exists()

    def test_analysis_caching(self, tmp_path):
        """Analysis should cache and reload from JSON."""
        from src.video_pipeline import analyse_video

        # Create minimal video
        video_path = tmp_path / "test.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (320, 240))
        for _ in range(10):
            out.write(np.zeros((240, 320, 3), dtype=np.uint8))
        out.release()

        cache_path = tmp_path / "cache.json"

        # First run
        analysis1 = analyse_video(video_path, cache_path=cache_path)
        assert cache_path.exists()

        # Second run should load from cache
        analysis2 = analyse_video(video_path, cache_path=cache_path)

        # Should be identical
        assert analysis1 == analysis2

    def test_no_detections_fallback(self, tmp_path):
        """Video with no faces should fall back to center crop."""
        from src.video_pipeline import analyse_video, render_reel

        # Create blank video (no faces)
        video_path = tmp_path / "blank.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (640, 480))
        for _ in range(25):
            out.write(np.full((480, 640, 3), 128, dtype=np.uint8))
        out.release()

        analysis = analyse_video(video_path, sample_fps=5.0)

        # Should have fallback crop path
        assert len(analysis["crop_path"]) > 0
        assert analysis["primary_track_id"] is None

        # Crop should be centered
        cp = analysis["crop_path"][0]
        assert abs(cp["focus_x"] - 0.5) < 0.01
        assert abs(cp["focus_y"] - 0.5) < 0.01

        # Should still render
        reel = render_reel(video_path, analysis, output_path=tmp_path / "reel.mp4")
        assert reel.exists()

    def test_crop_path_smoothing(self, tmp_path):
        """Crop path should be temporally smoothed."""
        from src.video_pipeline import _smooth_crop_path

        # Raw path with jitter
        raw = [
            {"frame": 0, "time": 0.0, "focus_x": 0.5, "focus_y": 0.5, "track_id": 0},
            {"frame": 5, "time": 0.2, "focus_x": 0.6, "focus_y": 0.5, "track_id": 0},
            {"frame": 10, "time": 0.4, "focus_x": 0.55, "focus_y": 0.5, "track_id": 0},
            {"frame": 15, "time": 0.6, "focus_x": 0.65, "focus_y": 0.5, "track_id": 0},
        ]

        smoothed = _smooth_crop_path(raw, window=3)

        # Smoothed should have less variance
        assert len(smoothed) == len(raw)
        # Middle point should be averaged
        assert abs(smoothed[2]["focus_x"] - 0.575) < 0.05


class TestVideoValidation:
    """Test video output validation."""

    def test_validate_video_basic(self, tmp_path):
        from src.validation import validate_video

        # Create a valid video
        video_path = tmp_path / "valid.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (1080, 1920))
        for _ in range(10):
            out.write(np.zeros((1920, 1080, 3), dtype=np.uint8))
        out.release()

        result = validate_video(video_path)

        assert result["ok"] is True
        assert any(c["name"] == "file_exists" for c in result["checks"])
        assert any(c["name"] == "openable" for c in result["checks"])
        assert any(c["name"] == "has_frames" for c in result["checks"])
