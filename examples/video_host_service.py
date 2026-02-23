"""Run host-side WebRTC signaling/media service for Quest video playback.

Example:
    uv run --with "hand-tracking-sdk[video]" python examples/video_host_service.py \
        --tcp-host 0.0.0.0 --tcp-port 8765 \
        --mocap-tcp-host 0.0.0.0 --mocap-tcp-port 8000 \
        --source webcam --webcam-index 0 --preset 720p30 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress

from hand_tracking_sdk.video.service import VideoService, VideoServiceConfig


class TelemetryTcpSink:
    """Minimal TCP sink for Quest->host mocap CSV telemetry."""

    def __init__(self, host: str, port: int, *, verbose: bool = False) -> None:
        self._host = host
        self._port = port
        self._verbose = verbose
        self._server: asyncio.AbstractServer | None = None
        self._client_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._on_client, self._host, self._port)
        print(f"[telemetry-tcp] listening host={self._host} port={self._port}")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for task in list(self._client_tasks):
            task.cancel()
        for task in list(self._client_tasks):
            with suppress(asyncio.CancelledError):
                await task
        self._client_tasks.clear()

    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)

        peer = writer.get_extra_info("peername")
        print(f"[telemetry-tcp] client connected remote={peer}")
        bytes_seen = 0
        lines_seen = 0

        try:
            while True:
                chunk = await reader.read(8192)
                if not chunk:
                    break
                bytes_seen += len(chunk)
                lines_seen += chunk.count(b"\n")
                if self._verbose and lines_seen and lines_seen % 200 == 0:
                    print(
                        "[telemetry-tcp]"
                        f" remote={peer} lines={lines_seen} bytes={bytes_seen}"
                    )
        except Exception as exc:
            print(f"[telemetry-tcp] client error remote={peer} error={exc}")
        finally:
            print(
                "[telemetry-tcp] client disconnected"
                f" remote={peer} lines={lines_seen} bytes={bytes_seen}"
            )
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            if task is not None:
                self._client_tasks.discard(task)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host signaling/WebRTC video service.")
    parser.add_argument("--signaling-host", default="0.0.0.0", help="WebSocket bind host.")
    parser.add_argument("--signaling-port", type=int, default=8765, help="WebSocket bind port.")
    parser.add_argument(
        "--tcp-host",
        default=None,
        help="Alias for --signaling-host (WebSocket signaling runs on TCP).",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=None,
        help="Alias for --signaling-port (WebSocket signaling runs on TCP).",
    )
    parser.add_argument(
        "--source",
        choices=("test", "webcam"),
        default="test",
        help="Video source adapter.",
    )
    parser.add_argument(
        "--preset",
        default="720p30",
        choices=("720p30", "1080p30"),
        help="Video preset.",
    )
    parser.add_argument("--webcam-index", type=int, default=0, help="Webcam device index.")
    parser.add_argument(
        "--mocap-tcp-host",
        default="0.0.0.0",
        help="TCP host for Quest telemetry (AppManager TCP connection).",
    )
    parser.add_argument(
        "--mocap-tcp-port",
        type=int,
        default=8000,
        help="TCP port for Quest telemetry (must match Quest telemetry port).",
    )
    parser.add_argument(
        "--disable-mocap-tcp",
        action="store_true",
        help="Disable telemetry TCP sink listener.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print signaling and WebRTC handshake debug logs.",
    )
    return parser.parse_args()


async def _run() -> int:
    args = _parse_args()
    signaling_host = args.tcp_host or args.signaling_host
    signaling_port = args.tcp_port or args.signaling_port
    telemetry_sink = None if args.disable_mocap_tcp else TelemetryTcpSink(
        args.mocap_tcp_host,
        args.mocap_tcp_port,
        verbose=args.verbose,
    )

    service = VideoService(
        VideoServiceConfig(
            signaling_host=signaling_host,
            signaling_port=signaling_port,
            source=args.source,
            preset=args.preset,
            webcam_index=args.webcam_index,
            verbose=args.verbose,
        )
    )
    if telemetry_sink is not None:
        await telemetry_sink.start()
    await service.start()
    print(
        "video service started"
        f" host={signaling_host}"
        f" port={signaling_port}"
        f" source={args.source}"
        f" preset={args.preset}"
    )
    print(f"signaling endpoint (TCP/WebSocket): ws://<HOST_IP>:{signaling_port}")
    try:
        while True:
            await asyncio.sleep(0.5)
    except KeyboardInterrupt:
        print("stopping video service")
    finally:
        await service.stop()
        if telemetry_sink is not None:
            await telemetry_sink.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
