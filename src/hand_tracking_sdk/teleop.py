"""Reusable hand-to-robot teleop mapping logic.

This module provides simulator-agnostic utilities for mapping hand tracking
data to robot arm targets.  The pinch metric and grip mapping work
identically regardless of the downstream sim backend (MuJoCo, Isaac, etc.).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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


# ---------------------------------------------------------------------------
# Finger curl angles
# ---------------------------------------------------------------------------

# Landmark chains per finger — each chain is a sequence of joints from base
# to tip.  The angle at each *interior* joint is the angle between the
# incoming and outgoing bone vectors.
_FINGER_CHAINS: dict[str, tuple[JointName, ...]] = {
    "thumb": (
        JointName.WRIST,
        JointName.THUMB_METACARPAL,
        JointName.THUMB_PROXIMAL,
        JointName.THUMB_DISTAL,
        JointName.THUMB_TIP,
    ),
    "index": (
        JointName.WRIST,
        JointName.INDEX_PROXIMAL,
        JointName.INDEX_INTERMEDIATE,
        JointName.INDEX_DISTAL,
        JointName.INDEX_TIP,
    ),
    "middle": (
        JointName.WRIST,
        JointName.MIDDLE_PROXIMAL,
        JointName.MIDDLE_INTERMEDIATE,
        JointName.MIDDLE_DISTAL,
        JointName.MIDDLE_TIP,
    ),
    "ring": (
        JointName.WRIST,
        JointName.RING_PROXIMAL,
        JointName.RING_INTERMEDIATE,
        JointName.RING_DISTAL,
        JointName.RING_TIP,
    ),
    "little": (
        JointName.WRIST,
        JointName.LITTLE_PROXIMAL,
        JointName.LITTLE_INTERMEDIATE,
        JointName.LITTLE_DISTAL,
        JointName.LITTLE_TIP,
    ),
}


def _vec_sub(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _angle_between(
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> float:
    """Angle in radians between two 3-D vectors."""
    dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    len_u = math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    len_v = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    denom = len_u * len_v
    if denom == 0.0:
        return 0.0
    # Clamp for numerical safety.
    return math.acos(max(-1.0, min(1.0, dot / denom)))


def finger_curl_angles(
    frame: HandFrame,
    fingers: Sequence[str] | None = None,
) -> dict[str, tuple[float, ...]]:
    """Compute inter-joint bend angles along each finger chain.

    For a chain of *N* joints the function returns *N - 2* angles (one per
    interior joint).  Each angle is between the incoming and outgoing bone
    vectors at that joint, measured in radians.  A straight finger yields
    angles close to ``0``; a fully curled finger yields larger angles
    (up to ``π``).

    :param frame:
        Hand frame with landmark data.
    :param fingers:
        Subset of finger names to compute (default: all five).
        Valid names: ``"thumb"``, ``"index"``, ``"middle"``, ``"ring"``,
        ``"little"``.
    :returns:
        Dict mapping finger name to a tuple of angles in radians.
    """
    chains = _FINGER_CHAINS
    if fingers is not None:
        chains = {k: v for k, v in chains.items() if k in fingers}

    result: dict[str, tuple[float, ...]] = {}
    for name, chain in chains.items():
        positions = [frame.get_joint(j) for j in chain]
        angles: list[float] = []
        for i in range(1, len(positions) - 1):
            incoming = _vec_sub(positions[i], positions[i - 1])
            outgoing = _vec_sub(positions[i + 1], positions[i])
            angles.append(_angle_between(incoming, outgoing))
        result[name] = tuple(angles)
    return result
