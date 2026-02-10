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
    VisualizationFrame,
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
        "--no-spawn",
        action="store_true",
        help="Do not auto-spawn rerun viewer.",
    )
    parser.add_argument(
        "--wrist-radius",
        type=float,
        default=0.02,
        help="Rerun point radius for wrist markers (meters).",
    )
    parser.add_argument(
        "--landmark-radius",
        type=float,
        default=0.01,
        help="Rerun point radius for landmark markers (meters).",
    )
    parser.add_argument(
        "--no-right-handed-conversion",
        action="store_true",
        help="Disable Unity->right-handed conversion before visualization.",
    )
    parser.add_argument(
        "--background-color",
        default="18,22,30",
        help=(
            "Rerun 3D background RGB as comma-separated values "
            "(e.g. 18,22,30). Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--visualization-frame",
        choices=[value.value for value in VisualizationFrame],
        default=VisualizationFrame.FLU.value,
        help="Output frame convention for visualization points.",
    )
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    background_color = _parse_rgb_or_none(args.background_color)

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
            spawn=not args.no_spawn,
            wrist_radius=args.wrist_radius,
            landmark_radius=args.landmark_radius,
            convert_to_right_handed=not args.no_right_handed_conversion,
            background_color=background_color,
            visualization_frame=VisualizationFrame(args.visualization_frame),
        )
    )

    for event in _stream_events(client.iter_events()):
        visualizer.log_event(event)

    return 0


def _stream_events(events: Iterable[object]) -> Iterable[object]:
    """Yield events unchanged so script remains easy to extend for preprocessing."""
    yield from events


def _parse_rgb_or_none(value: str) -> tuple[int, int, int] | None:
    if value.lower() == "none":
        return None

    chunks = [chunk.strip() for chunk in value.split(",")]
    if len(chunks) != 3:
        msg = "--background-color expects 3 comma-separated integers or 'none'."
        raise ValueError(msg)

    rgb = tuple(int(chunk) for chunk in chunks)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError("--background-color values must be in range [0, 255].")
    return rgb


if __name__ == "__main__":
    raise SystemExit(_main())
