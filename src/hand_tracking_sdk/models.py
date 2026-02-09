from dataclasses import dataclass
from enum import StrEnum


class HandSide(StrEnum):
    LEFT = "Left"
    RIGHT = "Right"


class PacketType(StrEnum):
    WRIST = "wrist"
    LANDMARKS = "landmarks"


@dataclass(frozen=True, slots=True)
class WristPose:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    points: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class WristPacket:
    side: HandSide
    kind: PacketType
    data: WristPose


@dataclass(frozen=True, slots=True)
class LandmarksPacket:
    side: HandSide
    kind: PacketType
    data: HandLandmarks


ParsedPacket = WristPacket | LandmarksPacket
