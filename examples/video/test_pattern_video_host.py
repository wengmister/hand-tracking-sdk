"""Run host-side signaling/media service with a synthetic test pattern source."""

from __future__ import annotations

import argparse
import asyncio

from _common import run_video_service

from hand_tracking_sdk.video.service import VideoServiceConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host video service (test pattern source).")
    parser.add_argument("--tcp-host", default="0.0.0.0", help="WebSocket signaling bind host.")
    parser.add_argument("--tcp-port", type=int, default=8765, help="WebSocket signaling bind port.")
    parser.add_argument(
        "--mocap-tcp-host",
        default="0.0.0.0",
        help="Telemetry TCP host for Quest mocap stream.",
    )
    parser.add_argument(
        "--mocap-tcp-port",
        type=int,
        default=8000,
        help="Telemetry TCP port for Quest mocap stream.",
    )
    parser.add_argument(
        "--disable-mocap-tcp",
        action="store_true",
        help="Disable telemetry TCP sink listener.",
    )
    parser.add_argument(
        "--preset",
        default="720p30",
        choices=("480p30", "480p60", "720p30", "720p60", "1080p30"),
        help="Video preset.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    return parser.parse_args()


async def _run() -> int:
    args = _parse_args()
    config = VideoServiceConfig(
        signaling_host=args.tcp_host,
        signaling_port=args.tcp_port,
        source="test",
        preset=args.preset,
        verbose=args.verbose,
    )
    return await run_video_service(
        config,
        enable_mocap_tcp=not args.disable_mocap_tcp,
        mocap_tcp_host=args.mocap_tcp_host,
        mocap_tcp_port=args.mocap_tcp_port,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
