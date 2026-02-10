"""Stream assembled hand frames from HTS and print a concise summary.

Example:
    uv run python examples/stream_frames.py --transport tcp_server --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse

from hand_tracking_sdk import (
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream assembled hand frames from HTS.")
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
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. Use 0 to stream indefinitely.",
    )
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    max_frames = args.max_frames if args.max_frames > 0 else None

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode(args.transport),
            host=args.host,
            port=args.port,
            timeout_s=args.timeout,
            output=StreamOutput.FRAMES,
        )
    )

    emitted = 0
    for frame in client.iter_events():
        emitted += 1
        print(
            "frame"
            f" seq={frame.sequence_id}"
            f" side={frame.side.value}"
            f" frame_id={frame.frame_id}"
            f" recv_ts_ns={frame.recv_ts_ns}"
            f" wrist=({frame.wrist.x:.3f}, {frame.wrist.y:.3f}, {frame.wrist.z:.3f})"
            f" landmarks={len(frame.landmarks.points)}"
        )
        if max_frames is not None and emitted >= max_frames:
            break

    stats = client.get_stats()
    print(
        "done"
        f" frames_emitted={stats.frames_emitted}"
        f" packets_emitted={stats.packets_emitted}"
        f" parse_errors={stats.parse_errors}"
        f" dropped_lines={stats.dropped_lines}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
