"""Parsing helpers for HTS UTF-8 CSV packets."""

from hand_tracking_sdk.constants import LANDMARK_COUNT, LANDMARK_VALUE_COUNT, WRIST_VALUE_COUNT
from hand_tracking_sdk.exceptions import ParseError
from hand_tracking_sdk.models import (
    HandLandmarks,
    HandSide,
    LandmarksPacket,
    PacketType,
    ParsedPacket,
    WristPacket,
    WristPose,
)


def parse_line(line: str) -> ParsedPacket:
    """Parse one HTS CSV line into a typed packet object.

    The input line must use one of the supported labels:
    ``Left wrist:``, ``Right wrist:``, ``Left landmarks:``, or
    ``Right landmarks:``.

    :param line:
        Raw UTF-8 decoded line from HTS transport.
    :returns:
        A parsed packet instance for wrist or landmark data.
    :rtype:
        ParsedPacket
    :raises ParseError:
        If the line is empty, malformed, has unsupported labels, includes
        non-float values, or does not match expected value counts.
    """
    stripped = line.strip()
    if not stripped:
        raise ParseError("Empty line.")

    head, sep, tail = stripped.partition(":")
    if not sep:
        raise ParseError("Missing ':' separator.")

    label = head.strip()
    payload = _parse_floats(tail)

    side, kind = _parse_label(label)
    if kind == PacketType.WRIST:
        return _parse_wrist(side=side, values=payload)
    return _parse_landmarks(side=side, values=payload)


def _parse_label(label: str) -> tuple[HandSide, PacketType]:
    """Parse packet label into hand side and packet type.

    :param label:
        Label segment before ``:`` (for example ``"Right wrist"``).
    :returns:
        Parsed hand side and packet type tuple.
    :raises ParseError:
        If label format, side, or packet type is unsupported.
    """
    parts = label.split()
    if len(parts) != 2:
        raise ParseError(f"Invalid label: {label!r}")

    side_raw, kind_raw = parts

    try:
        side = HandSide(side_raw)
    except ValueError as exc:
        raise ParseError(f"Unsupported hand side: {side_raw!r}") from exc

    normalized_kind = kind_raw.lower()
    if normalized_kind == PacketType.WRIST.value:
        return side, PacketType.WRIST
    if normalized_kind == PacketType.LANDMARKS.value:
        return side, PacketType.LANDMARKS
    raise ParseError(f"Unsupported packet type: {kind_raw!r}")


def _parse_floats(payload: str) -> list[float]:
    """Parse comma-separated numeric payload into floats.

    :param payload:
        CSV payload segment after ``:``.
    :returns:
        Parsed float list with empty chunks removed.
    :raises ParseError:
        If any value cannot be parsed as ``float``.
    """
    chunks = [chunk.strip() for chunk in payload.split(",") if chunk.strip()]
    try:
        return [float(value) for value in chunks]
    except ValueError as exc:
        raise ParseError("Payload contains non-float values.") from exc


def _parse_wrist(side: HandSide, values: list[float]) -> WristPacket:
    """Validate and map wrist values into a typed packet.

    :param side:
        Hand side of the packet.
    :param values:
        Parsed float values expected to contain exactly 7 elements.
    :returns:
        Typed wrist packet.
    :raises ParseError:
        If value count does not match the wrist contract.
    """
    if len(values) != WRIST_VALUE_COUNT:
        raise ParseError(f"Wrist packet must contain {WRIST_VALUE_COUNT} values, got {len(values)}")

    pose = WristPose(*values)
    return WristPacket(side=side, kind=PacketType.WRIST, data=pose)


def _parse_landmarks(side: HandSide, values: list[float]) -> LandmarksPacket:
    """Validate and map landmark values into a typed packet.

    :param side:
        Hand side of the packet.
    :param values:
        Parsed float values expected to contain exactly 63 elements.
    :returns:
        Typed landmark packet with 21 ``(x, y, z)`` points.
    :raises ParseError:
        If value count does not match the landmarks contract.
    """
    if len(values) != LANDMARK_VALUE_COUNT:
        raise ParseError(
            f"Landmarks packet must contain {LANDMARK_VALUE_COUNT} values, got {len(values)}"
        )

    points = tuple(
        (values[i], values[i + 1], values[i + 2]) for i in range(0, LANDMARK_COUNT * 3, 3)
    )
    return LandmarksPacket(side=side, kind=PacketType.LANDMARKS, data=HandLandmarks(points=points))
