"""
Tests for the pieces of the dataset that don't require the actual
(large, access-gated) EchoNet files: EF class thresholds and clip
index sampling. Run these before touching real data to catch label
or indexing bugs early -- they're cheap and fast.
"""
import numpy as np
import pytest

from src.data.echonet_dataset import ef_to_class


class TestEFToClass:
    def test_reduced(self):
        assert ef_to_class(25.0, [40, 50]) == 0

    def test_boundary_is_inclusive_on_upper_class(self):
        # EF exactly at a threshold should count as meeting it (>=), matching
        # the clinical convention (EF>=50 is "normal").
        assert ef_to_class(40.0, [40, 50]) == 1
        assert ef_to_class(50.0, [40, 50]) == 2

    def test_normal(self):
        assert ef_to_class(65.0, [40, 50]) == 2

    def test_single_threshold(self):
        assert ef_to_class(35.0, [50]) == 0
        assert ef_to_class(55.0, [50]) == 1


class TestClipSampling:
    """Exercises the index-sampling logic directly (copied minimal version)
    to check it never reads out of bounds -- the real class needs an actual
    cache directory to instantiate, so this checks the algorithm in isolation."""

    @staticmethod
    def sample_indices(total_frames, num_frames, stride, start):
        span = num_frames * stride
        if total_frames <= span:
            return (np.arange(num_frames) * stride) % total_frames
        return start + np.arange(num_frames) * stride

    def test_short_video_does_not_exceed_bounds(self):
        idx = self.sample_indices(total_frames=20, num_frames=32, stride=2, start=0)
        assert idx.max() < 20
        assert len(idx) == 32

    def test_normal_video_indices_in_bounds(self):
        total_frames = 150
        num_frames, stride, start = 32, 2, 10
        idx = self.sample_indices(total_frames, num_frames, stride, start)
        assert idx.min() >= 0
        assert idx.max() < total_frames
        assert len(idx) == num_frames
