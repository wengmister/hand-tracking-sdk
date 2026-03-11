"""Shared runner utilities for host-side video example scripts."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from threading import Thread
from typing import Any

from hand_tracking_sdk.client import (
    ErrorPolicy,
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)
from hand_tracking_sdk.frame import HandFrame, HeadFrame
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


def start_mocap_pump(
    host: str,
    port: int,
) -> dict[str, HandFrame | HeadFrame]:
    """Start a background thread that ingests mocap telemetry via HTSClient.

    Returns a shared dict keyed by ``"Left"`` / ``"Right"`` / ``"Head"``
    whose values are the latest frames.  The dict is updated from a daemon
    thread so reads from the MuJoCo render thread are lock-free (dict
    value assignment is atomic in CPython).
    """
    latest: dict[str, HandFrame | HeadFrame] = {}

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host=host,
            port=port,
            output=StreamOutput.FRAMES,
            error_policy=ErrorPolicy.TOLERANT,
        )
    )

    def _pump() -> None:
        for event in client.iter_events():
            if isinstance(event, (HandFrame, HeadFrame)):
                latest[event.side.value] = event

    thread = Thread(target=_pump, daemon=True)
    thread.start()
    return latest


def compensate_gravity(
    model: Any,
    data: Any,
    subtree_ids: list[int],
) -> None:
    """Apply gravity compensation forces to the given subtrees.

    Works for any MuJoCo model with position-actuated arms.  Imports
    ``mujoco`` and ``numpy`` lazily so the module stays importable without
    sim dependencies.
    """
    import mujoco
    import numpy as np

    data.qfrc_applied[:] = 0.0
    jac = np.empty((3, model.nv))
    for sid in subtree_ids:
        total_mass = model.body_subtreemass[sid]
        mujoco.mj_jacSubtreeCom(model, data, jac, sid)
        data.qfrc_applied[:] -= (model.opt.gravity * total_mass) @ jac


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


def build_perf_hook(interval: int = 60) -> Any:
    """Build a perf_hook that logs averaged timing every *interval* frames."""
    accum: dict[str, float] = {}
    count = 0

    def hook(metrics: dict[str, float]) -> None:
        nonlocal accum, count
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        count += 1
        if count >= interval:
            avg = {k: v / count for k, v in accum.items()}
            print(
                f"[perf] avg over {count} frames: "
                f"pre_step={avg.get('pre_step_ms', 0):.1f}ms "
                f"physics={avg.get('physics_ms', 0):.1f}ms "
                f"render={avg.get('render_ms', 0):.1f}ms "
                f"total={avg.get('total_ms', 0):.1f}ms "
                f"interval={avg.get('frame_interval_ms', 0):.1f}ms "
                f"steps={avg.get('n_physics_steps', 0):.0f}"
            )
            accum.clear()
            count = 0

    return hook


# ---------------------------------------------------------------------------
# Shared argument parsing
# ---------------------------------------------------------------------------


def build_base_parser(
    description: str,
    *,
    mujoco: bool = False,
    default_mj_model: str | None = None,
    default_mj_camera: str | None = None,
    default_preset: str = "720p",
    default_mocap_port: int = 8000,
) -> argparse.ArgumentParser:
    """Create an argument parser with standard video host arguments.

    When *mujoco* is ``True``, adds ``--mj-model``, ``--mj-camera``, and
    ``--perf`` arguments.
    """
    parser = argparse.ArgumentParser(description=description)
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
        default=default_mocap_port,
        help="Telemetry TCP port for Quest mocap stream.",
    )
    parser.add_argument(
        "--disable-mocap-tcp",
        action="store_true",
        help="Disable telemetry TCP listener.",
    )
    parser.add_argument(
        "--preset",
        default=default_preset,
        choices=("480p", "720p", "1080p"),
        help="Video resolution preset.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    if mujoco:
        parser.add_argument(
            "--mj-model", default=default_mj_model, help="Path to MuJoCo XML model."
        )
        parser.add_argument(
            "--mj-camera", default=default_mj_camera, help="MuJoCo camera name or id string."
        )
        parser.add_argument("--perf", action="store_true", help="Log per-frame timing breakdown.")
    return parser


# ---------------------------------------------------------------------------
# MuJoCo host runner
# ---------------------------------------------------------------------------

PreStepBuilder = Callable[[dict[str, HandFrame | HeadFrame], argparse.Namespace], Any]


async def run_mujoco_host(
    args: argparse.Namespace,
    pre_step_builder: PreStepBuilder | None = None,
) -> int:
    """Run a MuJoCo-based video host with standard mocap and perf wiring.

    *pre_step_builder* receives the mocap ``latest`` dict and parsed
    *args*, and returns a ``pre_step(model, data)`` callback.
    """
    pre_step = None
    if not args.disable_mocap_tcp and pre_step_builder is not None:
        latest = start_mocap_pump(args.mocap_tcp_host, args.mocap_tcp_port)
        pre_step = pre_step_builder(latest, args)

    perf_hook = build_perf_hook() if args.perf else None

    config = VideoServiceConfig(
        signaling_host=args.tcp_host,
        signaling_port=args.tcp_port,
        source="mujoco",
        preset=args.preset,
        mj_model_path=args.mj_model,
        mj_camera=args.mj_camera,
        mj_pre_step=pre_step,
        mj_perf_hook=perf_hook,
        verbose=args.verbose,
    )
    return await run_video_service(config, enable_mocap_tcp=False)
