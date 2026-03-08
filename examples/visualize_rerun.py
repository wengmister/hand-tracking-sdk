"""Run real-time hand telemetry visualization in rerun.

Example:
    uv run --with rerun-sdk python examples/visualize_rerun.py \\
        --transport tcp_server --host 0.0.0.0 --port 8000

To include optional head frame visualization:
    uv run --with rerun-sdk python examples/visualize_rerun.py \\
        --transport tcp_server --host 0.0.0.0 --port 8000 --output frames

To render pose coordinate frames (head/wrist):
    uv run --with rerun-sdk python examples/visualize_rerun.py \\
        --transport tcp_server --host 0.0.0.0 --port 8000 --show-coordinate-frames
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from hand_tracking_sdk import (
    ErrorPolicy,
    HandFilter,
    HTSClient,
    HTSClientConfig,
    RerunVisualizer,
    RerunVisualizerConfig,
    StreamEvent,
    StreamOutput,
    TransportMode,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize HTS telemetry in rerun.")
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
        "--reconnect-delay",
        type=float,
        default=0.25,
        help="Reconnect delay in seconds for TCP client mode.",
    )
    parser.add_argument(
        "--output",
        choices=[output.value for output in StreamOutput],
        default=StreamOutput.BOTH.value,
        help="Client output mode used for visualization input.",
    )
    parser.add_argument(
        "--hand-filter",
        choices=[value.value for value in HandFilter],
        default=HandFilter.BOTH.value,
        help="Filter for emitted hand side.",
    )
    parser.add_argument(
        "--error-policy",
        choices=[value.value for value in ErrorPolicy],
        default=ErrorPolicy.TOLERANT.value,
        help="Line parse behavior.",
    )
    parser.add_argument(
        "--application-id",
        default="hand-tracking-sdk",
        help="Rerun application id.",
    )
    parser.add_argument(
        "--show-jitter",
        action="store_true",
        default=False,
        help="Enable jitter/drop scalar timeseries panel.",
    )
    parser.add_argument(
        "--jitter-window-size",
        type=int,
        default=200,
        help="Rolling window size for jitter percentile metrics.",
    )
    parser.add_argument(
        "--show-coordinate-frames",
        action="store_true",
        default=False,
        help="Render local XYZ frame axes for wrist and head poses.",
    )
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode(args.transport),
            host=args.host,
            port=args.port,
            timeout_s=args.timeout,
            reconnect_delay_s=args.reconnect_delay,
            output=StreamOutput(args.output),
            hand_filter=HandFilter(args.hand_filter),
            error_policy=ErrorPolicy(args.error_policy),
        )
    )
    visualizer = RerunVisualizer(
        RerunVisualizerConfig(
            application_id=args.application_id,
            show_jitter_panel=args.show_jitter,
            jitter_window_size=args.jitter_window_size,
            show_coordinate_frames=args.show_coordinate_frames,
        )
    )

    for event in _stream_events(client.iter_events()):
        visualizer.log_event(event)

    return 0


def _stream_events(events: Iterable[StreamEvent]) -> Iterable[StreamEvent]:
    """Yield events unchanged so script remains easy to extend for preprocessing."""
    yield from events


if __name__ == "__main__":
    raise SystemExit(_main())
