"""Run real-time hand telemetry visualization in rerun.

Example:
    uv run --with rerun-sdk python examples/visualize_rerun.py \\
        --transport tcp_server --host 0.0.0.0 --port 8000
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
        )
    )

    for event in _stream_events(client.iter_events()):
        visualizer.log_event(event)

    return 0


def _stream_events(events: Iterable[object]) -> Iterable[object]:
    """Yield events unchanged so script remains easy to extend for preprocessing."""
    yield from events


if __name__ == "__main__":
    raise SystemExit(_main())
