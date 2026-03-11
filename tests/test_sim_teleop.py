"""Tests for MujocoSourceAdapter pre_step hook."""

from __future__ import annotations

from unittest.mock import MagicMock

from hand_tracking_sdk.video.service import VideoServiceConfig
from hand_tracking_sdk.video.sources import MujocoSourceAdapter


def test_pre_step_stored_on_adapter() -> None:
    """Verify pre_step callback is accepted and stored."""
    callback = MagicMock()
    adapter = MujocoSourceAdapter(model_path="dummy.xml", pre_step=callback)
    assert adapter._pre_step is callback


def test_pre_step_none_by_default() -> None:
    """Verify pre_step defaults to None when not provided."""
    adapter = MujocoSourceAdapter(model_path="dummy.xml")
    assert adapter._pre_step is None


def test_config_threads_mj_pre_step() -> None:
    """Verify VideoServiceConfig accepts mj_pre_step field."""
    callback = MagicMock()
    config = VideoServiceConfig(
        source="mujoco",
        mj_model_path="dummy.xml",
        mj_pre_step=callback,
    )
    assert config.mj_pre_step is callback
