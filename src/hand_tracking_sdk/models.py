"""Typed packet models for HTS telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hand_tracking_sdk._compat import StrEnum
from hand_tracking_sdk.constants import STREAMED_JOINT_NAMES


class HandSide(StrEnum):
    """Logical side for a tracked hand."""

    LEFT = "Left"
    RIGHT = "Right"


class PacketType(StrEnum):
    """Packet data category emitted by HTS."""

    WRIST = "wrist"
    LANDMARKS = "landmarks"


class JointName(StrEnum):
    """Canonical joint names matching HTS landmark order."""

    WRIST = "Wrist"
    THUMB_METACARPAL = "ThumbMetacarpal"
    THUMB_PROXIMAL = "ThumbProximal"
    THUMB_DISTAL = "ThumbDistal"
    THUMB_TIP = "ThumbTip"
    INDEX_PROXIMAL = "IndexProximal"
    INDEX_INTERMEDIATE = "IndexIntermediate"
    INDEX_DISTAL = "IndexDistal"
    INDEX_TIP = "IndexTip"
    MIDDLE_PROXIMAL = "MiddleProximal"
    MIDDLE_INTERMEDIATE = "MiddleIntermediate"
    MIDDLE_DISTAL = "MiddleDistal"
    MIDDLE_TIP = "MiddleTip"
    RING_PROXIMAL = "RingProximal"
    RING_INTERMEDIATE = "RingIntermediate"
    RING_DISTAL = "RingDistal"
    RING_TIP = "RingTip"
    LITTLE_PROXIMAL = "LittleProximal"
    LITTLE_INTERMEDIATE = "LittleIntermediate"
    LITTLE_DISTAL = "LittleDistal"
    LITTLE_TIP = "LittleTip"


class FingerName(StrEnum):
    """Supported finger groups for convenience accessors."""

    WRIST = "wrist"
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    LITTLE = "little"


_JOINT_INDEX_BY_NAME: dict[str, int] = {
    name: index for index, name in enumerate(STREAMED_JOINT_NAMES)
}


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

    def get_joint(self, joint: JointName | str) -> tuple[float, float, float]:
        """Return one joint point by name.

        :param joint:
            Joint to query, either as :class:`JointName` or canonical joint string
            (for example ``"IndexTip"``).
        :returns:
            Joint ``(x, y, z)`` tuple.
        :raises ValueError:
            If the joint name is unknown.
        """
        joint_name = joint.value if isinstance(joint, JointName) else joint
        index = _JOINT_INDEX_BY_NAME.get(joint_name)
        if index is None:
            raise ValueError(f"Unknown joint name: {joint_name!r}")
        return self.points[index]

    def get_finger(
        self,
        finger: FingerName | str,
    ) -> dict[JointName, tuple[float, float, float]]:
        """Return all joint points for one finger group.

        :param finger:
            Finger group to query. Accepts :class:`FingerName` or one of
            ``wrist``, ``thumb``, ``index``, ``middle``, ``ring``, ``little``.
        :returns:
            Dictionary mapping :class:`JointName` to ``(x, y, z)`` points for
            the selected finger group.
        :raises ValueError:
            If the finger group is unknown.
        """
        finger_name = finger.value if isinstance(finger, FingerName) else finger.lower()
        if finger_name == FingerName.WRIST.value:
            return {JointName.WRIST: self.get_joint(JointName.WRIST)}

        prefixes = {
            FingerName.THUMB.value: "Thumb",
            FingerName.INDEX.value: "Index",
            FingerName.MIDDLE.value: "Middle",
            FingerName.RING.value: "Ring",
            FingerName.LITTLE.value: "Little",
        }
        prefix = prefixes.get(finger_name)
        if prefix is None:
            raise ValueError(f"Unknown finger name: {finger_name!r}")

        result: dict[JointName, tuple[float, float, float]] = {}
        for joint in JointName:
            if joint is JointName.WRIST:
                continue
            if joint.value.startswith(prefix):
                result[joint] = self.get_joint(joint)
        return result

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
