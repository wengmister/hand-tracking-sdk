"""Run host-side signaling/media service with a synthetic test pattern source.

Usage::

    uv run examples/video/test_pattern_video_host.py
    uv run examples/video/test_pattern_video_host.py --preset 1080p --verbose
"""

from __future__ import annotations

import argparse
import asyncio

from _common import build_base_parser, run_video_service

from hand_tracking_sdk.video.service import VideoServiceConfig


def _parse_args() -> argparse.Namespace:
    return build_base_parser("Host video service (test pattern source).").parse_args()


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
