"""Smoke tests — verify imports and core geometry logic."""
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports():
    """All src packages import without error."""
    import src
    import src.utils
    import src.utils.geometry
    import src.detection
    import src.image_pipeline
    import src.video_pipeline
    import src.audio
    import src.validation
    import src.media
    import src.tracking


def test_compute_crop_box_center():
    from src.utils.geometry import compute_crop_box

    # 1920x1080 image, 1:1 crop, center focus
    x1, y1, x2, y2 = compute_crop_box(1920, 1080, 1, 1, 0.5, 0.5)
    w = x2 - x1
    h = y2 - y1
    assert w == h == 1080, f"Expected 1080x1080, got {w}x{h}"
    assert x1 >= 0 and y1 >= 0


def test_compute_crop_box_wide():
    from src.utils.geometry import compute_crop_box

    # 1920x1080, 16:9 → should be full frame
    x1, y1, x2, y2 = compute_crop_box(1920, 1080, 16, 9, 0.5, 0.5)
    assert x2 - x1 == 1920
    assert y2 - y1 == 1080


def test_compute_crop_box_tall():
    from src.utils.geometry import compute_crop_box

    # 1920x1080, 9:16 → height-limited
    x1, y1, x2, y2 = compute_crop_box(1920, 1080, 9, 16, 0.5, 0.5)
    w = x2 - x1
    h = y2 - y1
    ratio = w / h
    assert abs(ratio - 9 / 16) < 0.01, f"Bad ratio: {ratio}"


def test_compute_crop_box_focus_offset():
    from src.utils.geometry import compute_crop_box

    # Focus at left edge
    x1, y1, x2, y2 = compute_crop_box(1920, 1080, 1, 1, 0.1, 0.5)
    assert x1 == 0, "Should clamp to left edge"


def test_scale_roi():
    from src.utils.geometry import scale_roi

    roi = (10, 20, 100, 200)
    scaled = scale_roi(roi, (480, 270), (1920, 1080))
    assert scaled == (40, 80, 400, 800)


def test_validate_image_missing():
    from src.validation import validate_image

    result = validate_image("/nonexistent/image.jpg")
    assert result["ok"] is False
    assert result["checks"][0]["name"] == "file_exists"


def test_validate_video_missing():
    from src.validation import validate_video

    result = validate_video("/nonexistent/video.mp4")
    assert result["ok"] is False


def test_faces_center_empty():
    from src.detection import faces_center

    fx, fy = faces_center([], 1920, 1080)
    assert fx == 0.5 and fy == 0.5


def test_config_constants():
    from src.utils import ASPECT_RATIOS, REEL_WIDTH, REEL_HEIGHT

    assert len(ASPECT_RATIOS) == 4
    assert REEL_WIDTH == 1080
    assert REEL_HEIGHT == 1920
