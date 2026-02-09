"""Typed packet models for HTS telemetry."""

from dataclasses import dataclass
from enum import StrEnum


class HandSide(StrEnum):
    """Logical side for a tracked hand."""

    LEFT = "Left"
    RIGHT = "Right"


class PacketType(StrEnum):
    """Packet data category emitted by HTS."""

    WRIST = "wrist"
    LANDMARKS = "landmarks"


@dataclass(frozen=True, slots=True)
class WristPose:
    """Cartesian wrist position and orientation quaternion."""

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """Ordered set of 21 hand landmarks as ``(x, y, z)`` points."""

    points: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class WristPacket:
    """Parsed wrist packet for one hand side."""

    side: HandSide
    kind: PacketType
    data: WristPose


@dataclass(frozen=True, slots=True)
class LandmarksPacket:
    """Parsed landmark packet for one hand side."""

    side: HandSide
    kind: PacketType
    data: HandLandmarks


ParsedPacket = WristPacket | LandmarksPacket
