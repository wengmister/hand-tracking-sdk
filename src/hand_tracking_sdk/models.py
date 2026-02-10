"""Typed packet models for HTS telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


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

    def to_dict(self) -> dict[str, float]:
        """Serialize wrist pose into a mapping-friendly dictionary.

        :returns:
            Deterministic dictionary with position and quaternion fields.
        """
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> WristPose:
        """Build :class:`WristPose` from serialized mapping data.

        :param values:
            Mapping containing ``x, y, z, qx, qy, qz, qw``.
        :returns:
            Parsed wrist pose instance.
        """
        return cls(
            x=float(values["x"]),
            y=float(values["y"]),
            z=float(values["z"]),
            qx=float(values["qx"]),
            qy=float(values["qy"]),
            qz=float(values["qz"]),
            qw=float(values["qw"]),
        )


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """Ordered set of 21 hand landmarks as ``(x, y, z)`` points."""

    points: tuple[tuple[float, float, float], ...]

    def to_dict(self) -> dict[str, list[list[float]]]:
        """Serialize landmarks into a mapping-friendly dictionary.

        :returns:
            Dictionary with ordered ``points`` list.
        """
        return {"points": [[x, y, z] for x, y, z in self.points]}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> HandLandmarks:
        """Build :class:`HandLandmarks` from serialized mapping data.

        :param values:
            Mapping containing ``points`` as nested coordinate lists.
        :returns:
            Parsed landmarks instance preserving point order.
        """
        raw_points = values["points"]
        parsed_points = tuple(
            (float(point[0]), float(point[1]), float(point[2])) for point in raw_points
        )
        return cls(points=parsed_points)


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
