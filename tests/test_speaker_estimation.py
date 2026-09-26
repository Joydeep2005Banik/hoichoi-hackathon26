"""Tests for Phase 4 active-speaker estimation and sidecars."""
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestSpeakerEstimation:
    """Test lightweight active-speaker estimator and mouth motion."""

    def test_extract_mouth_region(self):
        from src.tracking import extract_mouth_region

        # 480x640 frame with a face at (200, 100, 400, 300) -> 200x200 face
        frame = np.full((480, 640), 128, dtype=np.uint8)
        patch = extract_mouth_region(frame, (200, 100, 400, 300))

        assert patch is not None
        assert patch.shape == (24, 40)

    def test_mouth_motion_update(self):
        from src.tracking import Track

        patch1 = np.zeros((24, 40), dtype=np.uint8)
        patch2 = np.full((24, 40), 50, dtype=np.uint8)

        track = Track(0, (100, 100, 200, 200), 0, patch1)
        assert track.mouth_motions == [0.0]

        track.update((100, 100, 200, 200), 1, patch2)
        assert len(track.mouth_motions) == 2
        assert track.mouth_motions[1] == 50.0
        assert track.recent_mouth_motion == 25.0

    def test_speaker_estimator_single_subject(self):
        from src.tracking import Track, SpeakerEstimator

        estimator = SpeakerEstimator()
        track = Track(0, (200, 100, 400, 300), 0)

        # When speech is active
        active_id, conf, scores = estimator.step([track], audio_speech_score=0.8, frame_idx=0, img_w=1920, img_h=1080)
        assert active_id == 0
        assert conf >= 0.8
        assert 0 in scores

    def test_speaker_estimator_two_speakers_dialogue(self):
        from src.tracking import Track, SpeakerEstimator

        estimator = SpeakerEstimator(switch_threshold=0.2, hold_frames=5)

        t1 = Track(1, (100, 200, 300, 400), 0)  # Person 1 (left)
        t2 = Track(2, (1500, 200, 1700, 400), 0) # Person 2 (right)

        # Simulate Person 1 speaking (higher mouth motion)
        t1.mouth_motions = [15.0, 18.0, 20.0]
        t2.mouth_motions = [1.0, 0.5, 0.0]

        active_id, conf, scores = estimator.step([t1, t2], audio_speech_score=0.7, frame_idx=0, img_w=1920, img_h=1080)
        assert active_id == 1
        assert scores[1] > scores[2]

        # Simulate switch to Person 2 speaking
        t1.mouth_motions = [0.0, 0.0, 0.0]
        t2.mouth_motions = [25.0, 30.0, 35.0]

        # Advance frame past hold time
        for i in range(1, 10):
            active_id, conf, scores = estimator.step([t1, t2], audio_speech_score=0.8, frame_idx=i, img_w=1920, img_h=1080)

        assert active_id == 2
        assert scores[2] > scores[1]

    def test_sidecar_generation(self, tmp_path):
        from src.video_pipeline import analyse_video

        # Create minimal 2-second video
        video_path = tmp_path / "test.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (320, 240))
        for _ in range(50):
            out.write(np.full((240, 320, 3), 90, dtype=np.uint8))
        out.release()

        analysis = analyse_video(video_path, sample_fps=5.0)

        assert "speaker_segments" in analysis
        assert "speaker_segments_path" in analysis
        assert "crop_path_path" in analysis

        # Verify sidecars exist on disk
        assert Path(analysis["speaker_segments_path"]).exists()
        assert Path(analysis["crop_path_path"]).exists()
