"""Tests for the hand_tracking_sdk.teleop module."""

from __future__ import annotations

import math

from hand_tracking_sdk.frame import HandFrame
from hand_tracking_sdk.models import HandLandmarks, HandSide, WristPose
from hand_tracking_sdk.teleop import (
    ArmTarget,
    GripConfig,
    extract_arm_target,
    grip_value,
    pinch_distance,
)


def _make_frame(
    *,
    thumb_tip: tuple[float, float, float] = (0.0, 0.0, 0.0),
    index_tip: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wrist_pos: tuple[float, float, float] = (1.0, 2.0, 3.0),
    wrist_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> HandFrame:
    """Build a HandFrame with controllable thumb/index tip positions."""
    # 21 landmarks: indices 4 = THUMB_TIP, 8 = INDEX_TIP.
    points = [(0.0, 0.0, 0.0)] * 21
    points[4] = thumb_tip
    points[8] = index_tip
    return HandFrame(
        side=HandSide.LEFT,
        frame_id="test",
        wrist=WristPose(
            x=wrist_pos[0],
            y=wrist_pos[1],
            z=wrist_pos[2],
            qx=wrist_quat[0],
            qy=wrist_quat[1],
            qz=wrist_quat[2],
            qw=wrist_quat[3],
        ),
        landmarks=HandLandmarks(points=tuple(points)),
        sequence_id=0,
        recv_ts_ns=0,
        recv_time_unix_ns=None,
        source_ts_ns=None,
        wrist_recv_ts_ns=0,
        landmarks_recv_ts_ns=0,
    )


def test_pinch_distance_zero() -> None:
    """Pinch distance is zero when thumb and index tips coincide."""
    frame = _make_frame(thumb_tip=(0.1, 0.2, 0.3), index_tip=(0.1, 0.2, 0.3))
    assert pinch_distance(frame) == 0.0


def test_pinch_distance_known() -> None:
    """Pinch distance matches expected Euclidean distance."""
    frame = _make_frame(thumb_tip=(0.0, 0.0, 0.0), index_tip=(0.03, 0.04, 0.0))
    assert math.isclose(pinch_distance(frame), 0.05, rel_tol=1e-9)


def test_grip_value_fully_closed() -> None:
    """Pinch below close_dist maps to ctrl_min."""
    config = GripConfig(open_dist=0.06, close_dist=0.02, ctrl_min=0.002, ctrl_max=0.037)
    frame = _make_frame(thumb_tip=(0.0, 0.0, 0.0), index_tip=(0.01, 0.0, 0.0))
    assert grip_value(frame, config) == config.ctrl_min


def test_grip_value_fully_open() -> None:
    """Pinch above open_dist maps to ctrl_max."""
    config = GripConfig(open_dist=0.06, close_dist=0.02, ctrl_min=0.002, ctrl_max=0.037)
    frame = _make_frame(thumb_tip=(0.0, 0.0, 0.0), index_tip=(0.1, 0.0, 0.0))
    assert grip_value(frame, config) == config.ctrl_max


def test_grip_value_midpoint() -> None:
    """Pinch at midpoint of close/open range maps to midpoint of ctrl range."""
    config = GripConfig(open_dist=0.06, close_dist=0.02, ctrl_min=0.002, ctrl_max=0.037)
    mid_dist = (config.close_dist + config.open_dist) / 2.0
    frame = _make_frame(thumb_tip=(0.0, 0.0, 0.0), index_tip=(mid_dist, 0.0, 0.0))
    expected = (config.ctrl_min + config.ctrl_max) / 2.0
    assert math.isclose(grip_value(frame, config), expected, rel_tol=1e-9)


def test_grip_config_defaults() -> None:
    """Default GripConfig matches ALOHA gripper range."""
    config = GripConfig()
    assert config.ctrl_min == 0.002
    assert config.ctrl_max == 0.037
    assert config.open_dist == 0.06
    assert config.close_dist == 0.02


def test_extract_arm_target_position() -> None:
    """ArmTarget position matches HandFrame wrist position."""
    frame = _make_frame(wrist_pos=(0.5, -0.3, 0.1))
    target = extract_arm_target(frame)
    assert target.position == (0.5, -0.3, 0.1)


def test_extract_arm_target_orientation() -> None:
    """ArmTarget orientation matches HandFrame wrist quaternion (qw, qx, qy, qz)."""
    frame = _make_frame(wrist_quat=(0.1, 0.2, 0.3, 0.9))
    target = extract_arm_target(frame)
    assert target.orientation == (0.9, 0.1, 0.2, 0.3)


def test_extract_arm_target_grip() -> None:
    """ArmTarget grip matches grip_value output."""
    config = GripConfig()
    frame = _make_frame(thumb_tip=(0.0, 0.0, 0.0), index_tip=(0.04, 0.0, 0.0))
    target = extract_arm_target(frame, config)
    expected_grip = grip_value(frame, config)
    assert target.grip == expected_grip


def test_extract_arm_target_returns_arm_target() -> None:
    """extract_arm_target returns an ArmTarget instance."""
    frame = _make_frame()
    target = extract_arm_target(frame)
    assert isinstance(target, ArmTarget)
