"""Run host-side MuJoCo video service with optional mocap-driven teleop."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
from threading import Thread
from typing import Any

from _common import run_video_service

from hand_tracking_sdk.client import (
    ErrorPolicy,
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)
from hand_tracking_sdk.convert import convert_hand_frame_unity_left_to_right
from hand_tracking_sdk.frame import HandFrame, HeadFrame
from hand_tracking_sdk.models import JointName
from hand_tracking_sdk.video.service import VideoServiceConfig

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "assets", "dual_arm_teleop.xml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host video service (MuJoCo source).")
    parser.add_argument(
        "--mj-model",
        default=_DEFAULT_MODEL,
        help="Path to MuJoCo XML model (default: bundled dual-arm).",
    )
    parser.add_argument("--mj-camera", default=None, help="MuJoCo camera name or id string.")
    parser.add_argument("--tcp-host", default="0.0.0.0", help="WebSocket signaling bind host.")
    parser.add_argument(
        "--tcp-port", type=int, default=8765, help="WebSocket signaling bind port."
    )
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
        help="Disable telemetry TCP listener (sim runs without mocap input).",
    )
    parser.add_argument(
        "--preset",
        default="720p30",
        choices=("720p30", "1080p30"),
        help="Video preset.",
    )
    parser.add_argument(
        "--left-mocap-body",
        default="left_target",
        help="MuJoCo mocap body name for left wrist.",
    )
    parser.add_argument(
        "--right-mocap-body",
        default="right_target",
        help="MuJoCo mocap body name for right wrist.",
    )
    parser.add_argument(
        "--left-gripper-actuator",
        default="left_gripper",
        help="MuJoCo actuator name for left gripper.",
    )
    parser.add_argument(
        "--right-gripper-actuator",
        default="right_gripper",
        help="MuJoCo actuator name for right gripper.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Mocap ingestion via HTSClient in a daemon thread
# ---------------------------------------------------------------------------


def _start_mocap_pump(
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
            if isinstance(event, HandFrame):
                event = convert_hand_frame_unity_left_to_right(event)
            latest[event.side.value] = event

    thread = Thread(target=_pump, daemon=True)
    thread.start()
    return latest


# ---------------------------------------------------------------------------
# Mocap → MuJoCo pre_step wiring
# ---------------------------------------------------------------------------


def _pinch_distance(frame: HandFrame) -> float:
    """Compute Euclidean distance between thumb tip and index tip."""
    thumb = frame.get_joint(JointName.THUMB_TIP)
    index = frame.get_joint(JointName.INDEX_TIP)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(thumb, index, strict=True)))


def _build_pre_step(
    latest: dict[str, HandFrame | HeadFrame],
    *,
    left_mocap_body: str,
    right_mocap_body: str,
    left_gripper_actuator: str,
    right_gripper_actuator: str,
    grip_open_dist: float = 0.06,
    grip_close_dist: float = 0.02,
    grip_range: float = 0.03,
) -> Any:
    """Build a pre_step callback that applies mocap state to MuJoCo data.

    Wrist poses are written to ``data.mocap_pos`` / ``data.mocap_quat``.
    Grip is computed from thumb-to-index pinch distance and mapped to
    gripper actuator control values.
    """
    body_ids: dict[str, int] = {}
    actuator_ids: dict[str, int] = {}

    def _grip_value(frame: HandFrame) -> float:
        """Map pinch distance to gripper actuator value (0=closed, grip_range=open)."""
        dist = _pinch_distance(frame)
        t = max(0.0, min(1.0, (dist - grip_close_dist) / (grip_open_dist - grip_close_dist)))
        return t * grip_range

    def pre_step(model: Any, data: Any) -> None:
        # Lazy-resolve body and actuator ids on first call.
        if not body_ids:
            try:
                import mujoco

                body_ids[left_mocap_body] = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, left_mocap_body
                )
                body_ids[right_mocap_body] = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, right_mocap_body
                )
                actuator_ids[left_gripper_actuator] = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, left_gripper_actuator
                )
                actuator_ids[right_gripper_actuator] = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, right_gripper_actuator
                )
            except Exception as exc:
                print(f"[mujoco-host] failed to resolve MuJoCo names: {exc}")
                return

        left = latest.get("Left")
        right = latest.get("Right")

        if isinstance(left, HandFrame):
            bid = body_ids[left_mocap_body]
            data.mocap_pos[bid] = [left.wrist.x, left.wrist.y, left.wrist.z]
            data.mocap_quat[bid] = [left.wrist.qw, left.wrist.qx, left.wrist.qy, left.wrist.qz]
            aid = actuator_ids[left_gripper_actuator]
            data.ctrl[aid] = _grip_value(left)
            # Mirror to coupled finger.
            coupled_id = aid + 1
            if coupled_id < model.nu:
                data.ctrl[coupled_id] = data.ctrl[aid]

        if isinstance(right, HandFrame):
            bid = body_ids[right_mocap_body]
            data.mocap_pos[bid] = [right.wrist.x, right.wrist.y, right.wrist.z]
            data.mocap_quat[bid] = [
                right.wrist.qw,
                right.wrist.qx,
                right.wrist.qy,
                right.wrist.qz,
            ]
            aid = actuator_ids[right_gripper_actuator]
            data.ctrl[aid] = _grip_value(right)
            coupled_id = aid + 1
            if coupled_id < model.nu:
                data.ctrl[coupled_id] = data.ctrl[aid]

    return pre_step


async def _run() -> int:
    args = _parse_args()

    # Set up mocap ingestion and pre_step hook when mocap is enabled.
    pre_step = None
    if not args.disable_mocap_tcp:
        latest = _start_mocap_pump(args.mocap_tcp_host, args.mocap_tcp_port)
        pre_step = _build_pre_step(
            latest,
            left_mocap_body=args.left_mocap_body,
            right_mocap_body=args.right_mocap_body,
            left_gripper_actuator=args.left_gripper_actuator,
            right_gripper_actuator=args.right_gripper_actuator,
        )

    config = VideoServiceConfig(
        signaling_host=args.tcp_host,
        signaling_port=args.tcp_port,
        source="mujoco",
        preset=args.preset,
        mj_model_path=args.mj_model,
        mj_camera=args.mj_camera,
        mj_pre_step=pre_step,
        verbose=args.verbose,
    )
    return await run_video_service(config, enable_mocap_tcp=False)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
