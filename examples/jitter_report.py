"""Stream HTS frames and print rolling network jitter/drop metrics.

This script uses inter-frame timing deltas, not absolute source-vs-recv clock
difference, because source and receiver monotonic clocks are independent.

Example:
    uv run python examples/jitter_report.py --transport tcp_server --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from statistics import median

from hand_tracking_sdk import (
    HandFrame,
    HeadFrame,
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling HTS jitter/drop report by hand side.")
    parser.add_argument(
        "--transport",
        choices=[mode.value for mode in TransportMode],
        default=TransportMode.TCP_SERVER.value,
        help="Transport mode for inbound stream.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host bind/connect address.")
    parser.add_argument("--port", type=int, default=8000, help="Host bind/connect port.")
    parser.add_argument("--timeout", type=float, default=1.0, help="I/O timeout in seconds.")
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="Rolling window size in frames per hand side.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=50,
        help="Print one report line after this many new frames per hand side.",
    )
    return parser.parse_args()


@dataclass(slots=True)
class _SideMetrics:
    prev_recv_ts_ns: int | None = None
    prev_source_ts_ns: int | None = None
    prev_source_frame_seq: int | None = None
    jitter_ns_window: deque[int] = field(default_factory=deque)
    source_dt_ns_window: deque[int] = field(default_factory=deque)
    recv_dt_ns_window: deque[int] = field(default_factory=deque)
    dropped_frames_window: deque[int] = field(default_factory=deque)
    updates_since_report: int = 0


def _append_window(window: deque[int], value: int, maxlen: int) -> None:
    window.append(value)
    if len(window) > maxlen:
        window.popleft()


def _percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * fraction))
    return sorted_values[index]


def _main() -> int:
    args = _parse_args()
    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode(args.transport),
            host=args.host,
            port=args.port,
            timeout_s=args.timeout,
            output=StreamOutput.FRAMES,
        )
    )

    side_state: dict[str, _SideMetrics] = {}
    for event in client.iter_events():
        if not isinstance(event, (HandFrame, HeadFrame)):
            continue
        frame = event
        side = frame.side.value
        state = side_state.setdefault(side, _SideMetrics())

        if (
            frame.source_ts_ns is not None
            and state.prev_source_ts_ns is not None
            and state.prev_recv_ts_ns is not None
        ):
            source_dt_ns = frame.source_ts_ns - state.prev_source_ts_ns
            recv_dt_ns = frame.recv_ts_ns - state.prev_recv_ts_ns
            jitter_ns = recv_dt_ns - source_dt_ns

            _append_window(state.source_dt_ns_window, source_dt_ns, args.window)
            _append_window(state.recv_dt_ns_window, recv_dt_ns, args.window)
            _append_window(state.jitter_ns_window, jitter_ns, args.window)

        if frame.source_frame_seq is not None and state.prev_source_frame_seq is not None:
            gap = max(0, frame.source_frame_seq - state.prev_source_frame_seq - 1)
            _append_window(state.dropped_frames_window, gap, args.window)

        state.prev_recv_ts_ns = frame.recv_ts_ns
        state.prev_source_ts_ns = frame.source_ts_ns
        state.prev_source_frame_seq = frame.source_frame_seq
        state.updates_since_report += 1

        if state.updates_since_report < args.report_every:
            continue
        state.updates_since_report = 0

        if not state.jitter_ns_window:
            print(
                f"side={side} waiting for debug metadata "
                "(need source_ts_ns/source_frame_seq in stream)"
            )
            continue

        jitter_sorted = sorted(state.jitter_ns_window)
        drops_sum = sum(state.dropped_frames_window)
        source_rate_hz = (
            1e9 / median(state.source_dt_ns_window)
            if state.source_dt_ns_window and median(state.source_dt_ns_window) > 0
            else 0.0
        )
        recv_rate_hz = (
            1e9 / median(state.recv_dt_ns_window)
            if state.recv_dt_ns_window and median(state.recv_dt_ns_window) > 0
            else 0.0
        )

        print(
            f"side={side}"
            f" n={len(state.jitter_ns_window)}"
            f" src_rate_hz={source_rate_hz:.1f}"
            f" recv_rate_hz={recv_rate_hz:.1f}"
            f" jitter_p50_ms={_percentile(jitter_sorted, 0.50) / 1e6:.3f}"
            f" jitter_p95_ms={_percentile(jitter_sorted, 0.95) / 1e6:.3f}"
            f" jitter_p99_ms={_percentile(jitter_sorted, 0.99) / 1e6:.3f}"
            f" drops_window={drops_sum}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

