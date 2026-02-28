"""Reusable hand-to-robot teleop mapping logic.

This module provides simulator-agnostic utilities for mapping hand tracking
data to robot arm targets.  The pinch metric and grip mapping work
identically regardless of the downstream sim backend (MuJoCo, Isaac, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hand_tracking_sdk.frame import HandFrame
from hand_tracking_sdk.models import JointName


@dataclass(frozen=True, slots=True)
class GripConfig:
    """Pinch-to-grip mapping parameters.

    :param open_dist:
        Pinch distance (meters) at which gripper is fully open.
    :param close_dist:
        Pinch distance (meters) at which gripper is fully closed.
    :param ctrl_min:
        Actuator control value for fully closed gripper.
    :param ctrl_max:
        Actuator control value for fully open gripper.
    """

    open_dist: float = 0.06
    close_dist: float = 0.02
    ctrl_min: float = 0.002
    ctrl_max: float = 0.037


@dataclass(frozen=True, slots=True)
class ArmTarget:
    """Arm end-effector target extracted from a hand frame.

    :param position:
        Target EE position ``(x, y, z)`` in right-handed coordinates.
    :param orientation:
        Target EE orientation quaternion ``(qw, qx, qy, qz)``.
    :param grip:
        Gripper actuator control value mapped from pinch distance.
    """

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    grip: float


_DEFAULT_GRIP_CONFIG = GripConfig()


def pinch_distance(frame: HandFrame) -> float:
    """Compute Euclidean distance between thumb tip and index tip.

    :param frame:
        Hand frame with landmark data.
    :returns:
        Distance in the same units as the landmark coordinates.
    """
    thumb = frame.get_joint(JointName.THUMB_TIP)
    index = frame.get_joint(JointName.INDEX_TIP)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(thumb, index, strict=True)))


def grip_value(frame: HandFrame, config: GripConfig = _DEFAULT_GRIP_CONFIG) -> float:
    """Map pinch distance to gripper actuator control value.

    :param frame:
        Hand frame with landmark data.
    :param config:
        Grip mapping parameters.
    :returns:
        Actuator control value in ``[config.ctrl_min, config.ctrl_max]``.
    """
    dist = pinch_distance(frame)
    t = max(
        0.0,
        min(1.0, (dist - config.close_dist) / (config.open_dist - config.close_dist)),
    )
    return config.ctrl_min + t * (config.ctrl_max - config.ctrl_min)


def extract_arm_target(
    frame: HandFrame,
    grip_config: GripConfig = _DEFAULT_GRIP_CONFIG,
) -> ArmTarget:
    """Extract arm EE target pose and grip value from a hand frame.

    :param frame:
        Hand frame (expected to be in right-handed coordinates after
        coordinate conversion).
    :param grip_config:
        Grip mapping parameters.
    :returns:
        Arm target with position, orientation, and grip control value.
    """
    w = frame.wrist
    return ArmTarget(
        position=(w.x, w.y, w.z),
        orientation=(w.qw, w.qx, w.qy, w.qz),
        grip=grip_value(frame, grip_config),
    )
