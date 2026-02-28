"""Shared runner utilities for host-side video example scripts."""

from __future__ import annotations

import asyncio

from hand_tracking_sdk.video.service import VideoService, VideoServiceConfig


async def _run_telemetry_sink(
    host: str,
    port: int,
    verbose: bool = False,
) -> asyncio.AbstractServer:
    """Start a TCP server that accepts the Quest mocap connection and discards data.

    The VR app requires a listening TCP endpoint to transition into the
    streaming phase.  For non-simulation sources (webcam, test pattern)
    the telemetry data is not needed, so we just drain and discard it.
    """

    async def _handle(reader: asyncio.StreamReader, _writer: asyncio.StreamWriter) -> None:
        if verbose:
            print(f"[telemetry-sink] client connected on {host}:{port}")
        try:
            while not reader.at_eof():
                await reader.readline()
        except (ConnectionError, asyncio.CancelledError):
            pass

    server = await asyncio.start_server(_handle, host, port)
    if verbose:
        print(f"[telemetry-sink] listening on {host}:{port}")
    return server


async def run_video_service(
    config: VideoServiceConfig,
    *,
    enable_mocap_tcp: bool = True,
    mocap_tcp_host: str = "0.0.0.0",
    mocap_tcp_port: int = 8000,
) -> int:
    """Run one configured video service instance until interrupted.

    When *enable_mocap_tcp* is ``True`` (the default), a lightweight TCP
    server is started to accept the Quest mocap connection.  Set to
    ``False`` when the caller handles mocap ingestion separately (e.g.
    the MuJoCo example uses ``HTSClient`` in a daemon thread).
    """
    sink_server: asyncio.AbstractServer | None = None
    if enable_mocap_tcp:
        sink_server = await _run_telemetry_sink(
            mocap_tcp_host, mocap_tcp_port, verbose=config.verbose
        )

    service = VideoService(config)
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
        if sink_server is not None:
            sink_server.close()
            await sink_server.wait_closed()
    return 0
