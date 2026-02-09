import pytest

from hand_tracking_sdk import (
    HandSide,
    LandmarksPacket,
    PacketType,
    ParseError,
    WristPacket,
    parse_line,
)


def test_parse_wrist_packet() -> None:
    packet = parse_line("Right wrist:, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0")

    assert isinstance(packet, WristPacket)
    assert packet.side == HandSide.RIGHT
    assert packet.kind == PacketType.WRIST
    assert packet.data.qw == 1.0


def test_parse_landmarks_packet() -> None:
    values = ", ".join(str(i / 100.0) for i in range(63))
    packet = parse_line(f"Left landmarks:, {values}")

    assert isinstance(packet, LandmarksPacket)
    assert packet.side == HandSide.LEFT
    assert packet.kind == PacketType.LANDMARKS
    assert len(packet.data.points) == 21
    assert packet.data.points[0] == (0.0, 0.01, 0.02)


def test_parse_trailing_comma_and_spaces() -> None:
    packet = parse_line(" Left wrist: 1, 2, 3, 4, 5, 6, 7, ")

    assert isinstance(packet, WristPacket)
    assert packet.side == HandSide.LEFT
    assert packet.data.qw == 7.0


@pytest.mark.parametrize(
    "line",
    [
        "",
        "Left wrist 1,2,3",
        "Middle wrist:, 1,2,3,4,5,6,7",
        "Right unknown:, 1,2,3",
        "Right wrist:, 1,2,3",
        "Left landmarks:, 1,2,3",
        "Right wrist:, 1,2,3,4,5,6,foo",
    ],
)
def test_parse_invalid_lines_raise(line: str) -> None:
    with pytest.raises(ParseError):
        parse_line(line)
