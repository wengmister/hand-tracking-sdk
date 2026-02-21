from collections.abc import Iterator

import pytest

from hand_tracking_sdk import (
    ClientCallbackError,
    ClientConfigurationError,
    ErrorPolicy,
    HandFilter,
    HandFrame,
    HandSide,
    HTSClient,
    HTSClientConfig,
    LandmarksPacket,
    LogEventKind,
    ParseError,
    StreamLogEvent,
    StreamOutput,
    WristPacket,
)


class FakeLineReceiver:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> "FakeLineReceiver":
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


def test_stats_track_parse_and_filter_drops() -> None:
    lines = [
        "bad line",
        "Right wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
        "Right landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
        "Left wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
        "Left landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
    ]
    client = _make_client(
        HTSClientConfig(
            output=StreamOutput.FRAMES,
            hand_filter=HandFilter.LEFT,
            error_policy=ErrorPolicy.TOLERANT,
        ),
        lines,
    )

    events = list(client.iter_events())
    stats = client.get_stats()

    assert len(events) == 1
    assert stats.lines_received == 5
    assert stats.parse_errors == 1
    assert stats.packets_filtered == 2
    assert stats.dropped_lines == 3
    assert stats.frames_emitted == 1


def test_structured_log_hook_receives_events() -> None:
    events: list[StreamLogEvent] = []
    lines = [
        "Left wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
        "Left landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
    ]
    client = _make_client(
        HTSClientConfig(output=StreamOutput.FRAMES, log_hook=events.append),
        lines,
    )

    output = list(client.iter_events())

    assert len(output) == 1
    assert any(event.kind == LogEventKind.RECEIVED_LINE for event in events)
    assert any(event.kind == LogEventKind.EMITTED_FRAME for event in events)


def test_callback_error_can_be_wrapped() -> None:
    lines = [
        "Left wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
        "Left landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
    ]
    client = _make_client(HTSClientConfig(output=StreamOutput.FRAMES), lines)

    def _bad_callback(_: object) -> None:
        raise RuntimeError("boom")

    with pytest.raises(ClientCallbackError):
        client.run(_bad_callback, wrap_callback_exceptions=True)

    stats = client.get_stats()
    assert stats.callback_errors == 1


def test_invalid_client_config_raises() -> None:
    with pytest.raises(ClientConfigurationError):
        HTSClientConfig(port=-1)
