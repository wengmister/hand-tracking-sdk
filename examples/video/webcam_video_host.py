"""Run host-side signaling/media service with a USB webcam source.

Usage::

    uv run examples/video/webcam_video_host.py
    uv run examples/video/webcam_video_host.py --webcam-index 1 --preset 1080p
"""

from __future__ import annotations

import argparse
import asyncio

from _common import build_base_parser, run_video_service

from hand_tracking_sdk.video.service import VideoServiceConfig


def _parse_args() -> argparse.Namespace:
    parser = build_base_parser("Host video service (webcam source).")
    parser.add_argument("--webcam-index", type=int, default=0, help="Webcam device index.")
    return parser.parse_args()


async def _run() -> int:
    args = _parse_args()
    config = VideoServiceConfig(
        signaling_host=args.tcp_host,
        signaling_port=args.tcp_port,
        source="webcam",
        preset=args.preset,
        webcam_index=args.webcam_index,
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
