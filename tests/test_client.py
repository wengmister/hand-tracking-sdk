from collections.abc import Iterator
from typing import Self

import pytest

from hand_tracking_sdk import (
    ErrorPolicy,
    HandFilter,
    HandFrame,
    HandSide,
    HTSClient,
    HTSClientConfig,
    LandmarksPacket,
    ParseError,
    StreamOutput,
    WristPacket,
)


class FakeLineReceiver:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def iter_lines(self) -> Iterator[str]:
        yield from self._lines


def _make_client(config: HTSClientConfig, lines: list[str]) -> HTSClient:
    return HTSClient(config, receiver_factory=lambda _: FakeLineReceiver(lines))


def test_iter_events_frames_mode_emits_assembled_frames() -> None:
    lines = [
        "Right wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
        "Right landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
    ]
    client = _make_client(
        HTSClientConfig(output=StreamOutput.FRAMES, hand_filter=HandFilter.BOTH),
        lines,
    )

    events = list(client.iter_events())

    assert len(events) == 1
    assert isinstance(events[0], HandFrame)
    assert events[0].side == HandSide.RIGHT


def test_iter_events_packets_mode_emits_packets_only() -> None:
    lines = [
        "Left wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
        "Left landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
    ]
    client = _make_client(HTSClientConfig(output=StreamOutput.PACKETS), lines)

    events = list(client.iter_events())

    assert len(events) == 2
    assert isinstance(events[0], WristPacket)
    assert isinstance(events[1], LandmarksPacket)


def test_iter_events_both_mode_emits_packets_and_frames() -> None:
    lines = [
        "Left wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
        "Left landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
    ]
    client = _make_client(HTSClientConfig(output=StreamOutput.BOTH), lines)

    events = list(client.iter_events())

    assert len(events) == 3
    assert isinstance(events[-1], HandFrame)


def test_hand_filter_left_drops_right() -> None:
    lines = [
        "Right wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
        "Right landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
        "Left wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
        "Left landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
    ]
    client = _make_client(
        HTSClientConfig(output=StreamOutput.FRAMES, hand_filter=HandFilter.LEFT),
        lines,
    )

    events = list(client.iter_events())

    assert len(events) == 1
    assert isinstance(events[0], HandFrame)
    assert events[0].side == HandSide.LEFT


def test_strict_mode_raises_parse_error() -> None:
    client = _make_client(
        HTSClientConfig(error_policy=ErrorPolicy.STRICT),
        ["bad line"],
    )

    with pytest.raises(ParseError):
        list(client.iter_events())


def test_tolerant_mode_skips_bad_lines() -> None:
    lines = [
        "bad line",
        "Right wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
        "Right landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
    ]
    client = _make_client(
        HTSClientConfig(output=StreamOutput.FRAMES, error_policy=ErrorPolicy.TOLERANT),
        lines,
    )

    events = list(client.iter_events())

    assert len(events) == 1
    assert isinstance(events[0], HandFrame)


def test_run_callback_honors_max_events() -> None:
    lines = [
        "Left wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
        "Left landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
    ]
    client = _make_client(HTSClientConfig(output=StreamOutput.BOTH), lines)

    seen: list[object] = []
    count = client.run(seen.append, max_events=2)

    assert count == 2
    assert len(seen) == 2
