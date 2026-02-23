"""Shared runner utilities for host-side video example scripts."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from hand_tracking_sdk.video.service import VideoService, VideoServiceConfig


class TelemetryTcpSink:
    """Minimal TCP sink for Quest->host mocap CSV telemetry."""

    def __init__(self, host: str, port: int, *, verbose: bool = False) -> None:
        """Initialize telemetry listener configuration."""
        self._host = host
        self._port = port
        self._verbose = verbose
        self._server: asyncio.AbstractServer | None = None
        self._client_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        """Start listening for telemetry TCP client connections."""
        self._server = await asyncio.start_server(self._on_client, self._host, self._port)
        print(f"[telemetry-tcp] listening host={self._host} port={self._port}")

    async def stop(self) -> None:
        """Stop telemetry listener and active client tasks."""
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


async def run_video_service(
    config: VideoServiceConfig,
    *,
    enable_mocap_tcp: bool = True,
    mocap_tcp_host: str = "0.0.0.0",
    mocap_tcp_port: int = 8000,
) -> int:
    """Run one configured video service instance until interrupted."""
    telemetry_sink = None
    if enable_mocap_tcp:
        telemetry_sink = TelemetryTcpSink(
            mocap_tcp_host,
            mocap_tcp_port,
            verbose=config.verbose,
        )

    service = VideoService(config)
    if telemetry_sink is not None:
        await telemetry_sink.start()

    await service.start()
    print(
        "video service started"
        f" host={config.signaling_host}"
        f" port={config.signaling_port}"
        f" source={config.source}"
        f" preset={config.preset}"
    )
    print(f"signaling endpoint (TCP/WebSocket): ws://<HOST_IP>:{config.signaling_port}")

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
