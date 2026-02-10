"""Stream HTS events and persist them as JSON Lines.

Example:
    uv run python examples/log_to_jsonl.py --transport tcp_server \\
        --host 0.0.0.0 --port 8000 --output both --path runs/hts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time_ns
from typing import Any

from hand_tracking_sdk import (
    HandFrame,
    HTSClient,
    HTSClientConfig,
    LandmarksPacket,
    StreamOutput,
    TransportMode,
    WristPacket,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log HTS stream events to JSONL.")
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
        "--output",
        choices=[value.value for value in StreamOutput],
        default=StreamOutput.BOTH.value,
        help="Event output type to log.",
    )
    parser.add_argument(
        "--path",
        default="runs/hand_tracking.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop after N events. Use 0 to stream indefinitely.",
    )
    return parser.parse_args()


def _event_to_dict(event: HandFrame | WristPacket | LandmarksPacket) -> dict[str, Any]:
    if isinstance(event, HandFrame):
        return {
            "event_type": "frame",
            "logged_at_unix_ns": time_ns(),
            "data": event.to_dict(),
        }
    if isinstance(event, WristPacket):
        return {
            "event_type": "packet",
            "packet_type": "wrist",
            "side": event.side.value,
            "logged_at_unix_ns": time_ns(),
            "data": event.data.to_dict(),
        }
    return {
        "event_type": "packet",
        "packet_type": "landmarks",
        "side": event.side.value,
        "logged_at_unix_ns": time_ns(),
        "data": event.data.to_dict(),
    }


def _main() -> int:
    args = _parse_args()
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_events = args.max_events if args.max_events > 0 else None

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode(args.transport),
            host=args.host,
            port=args.port,
            timeout_s=args.timeout,
            output=StreamOutput(args.output),
        )
    )

    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for event in client.iter_events():
            payload = _event_to_dict(event)
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            written += 1
            if max_events is not None and written >= max_events:
                break

    stats = client.get_stats()
    print(
        f"wrote {written} event(s) to {path}"
        f" frames_emitted={stats.frames_emitted}"
        f" packets_emitted={stats.packets_emitted}"
        f" parse_errors={stats.parse_errors}"
        f" dropped_lines={stats.dropped_lines}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
