"""Run Inspire bimanual host with lightweight vector retargeting.

This variant keeps the demo structure from other video hosts, but replaces
hand-crafted finger curl mapping with vector-based optimization from
``_retarget.MujocoVectorRetargeter``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

import numpy as np

from _common import build_perf_hook, run_video_service, start_mocap_pump
from _retarget import MujocoVectorRetargeter, default_tasks
from _tracking import RelativeHeadCamera, RelativeWristTracker

from hand_tracking_sdk.convert import BASIS_UNITY_LEFT_TO_RFU
from hand_tracking_sdk.frame import HandFrame, HeadFrame
from hand_tracking_sdk.models import JointName
from hand_tracking_sdk.video.service import VideoServiceConfig

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "assets", "inspire", "scene_bimanual.xml")

# Unity left-handed (x right, y up, z forward) -> MuJoCo (x right, y forward, z up).
_INSPIRE_BASIS = BASIS_UNITY_LEFT_TO_RFU

_SIDE_CONFIG: dict[str, dict[str, Any]] = {
    "left": {
        "frame_key": "Left",
        "base_actuators": (
            "left_pos_x_position",
            "left_pos_y_position",
            "left_pos_z_position",
        ),
        "ball_joint": "left_rot",
        "finger_joints": [
            "left_thumb_proximal_yaw_joint",
            "left_thumb_proximal_pitch_joint",
            "left_thumb_intermediate_joint",
            "left_thumb_distal_joint",
            "left_index_proximal_joint",
            "left_index_intermediate_joint",
            "left_middle_proximal_joint",
            "left_middle_intermediate_joint",
            "left_ring_proximal_joint",
            "left_ring_intermediate_joint",
            "left_pinky_proximal_joint",
            "left_pinky_intermediate_joint",
        ],
        "site_by_joint": {
            JointName.WRIST: "left_palm",
            JointName.THUMB_TIP: "left_thumb_tip",
            JointName.INDEX_TIP: "left_index_tip",
            JointName.MIDDLE_TIP: "left_middle_tip",
            JointName.RING_TIP: "left_ring_tip",
            JointName.LITTLE_TIP: "left_pinky_tip",
        },
    },
    "right": {
        "frame_key": "Right",
        "base_actuators": (
            "right_pos_x_position",
            "right_pos_y_position",
            "right_pos_z_position",
        ),
        "ball_joint": "right_rot",
        "finger_joints": [
            "right_thumb_proximal_yaw_joint",
            "right_thumb_proximal_pitch_joint",
            "right_thumb_intermediate_joint",
            "right_thumb_distal_joint",
            "right_index_proximal_joint",
            "right_index_intermediate_joint",
            "right_middle_proximal_joint",
            "right_middle_intermediate_joint",
            "right_ring_proximal_joint",
            "right_ring_intermediate_joint",
            "right_pinky_proximal_joint",
            "right_pinky_intermediate_joint",
        ],
        "site_by_joint": {
            JointName.WRIST: "right_palm",
            JointName.THUMB_TIP: "right_thumb_tip",
            JointName.INDEX_TIP: "right_index_tip",
            JointName.MIDDLE_TIP: "right_middle_tip",
            JointName.RING_TIP: "right_ring_tip",
            JointName.LITTLE_TIP: "right_pinky_tip",
        },
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host video service (Inspire vector retarget).")
    parser.add_argument(
        "--mj-model",
        default=_DEFAULT_MODEL,
        help="Path to MuJoCo XML model (default: bundled Inspire bimanual).",
    )
    parser.add_argument("--mj-camera", default="overview", help="MuJoCo camera name or id string.")
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
        default=5555,
        help="Telemetry TCP port for Quest mocap stream.",
    )
    parser.add_argument(
        "--disable-mocap-tcp",
        action="store_true",
        help="Disable telemetry TCP listener (sim runs without mocap input).",
    )
    parser.add_argument(
        "--preset",
        default="480p",
        choices=("480p", "720p", "1080p"),
        help="Video resolution preset (default 480p).",
    )
    parser.add_argument("--retarget-iters", type=int, default=5, help="GN iterations per frame.")
    parser.add_argument("--retarget-damping", type=float, default=1e-3, help="LM damping value.")
    parser.add_argument("--retarget-step", type=float, default=0.5, help="GN step size.")
    parser.add_argument("--retarget-tol", type=float, default=1e-4, help="Early-stop dq norm.")
    parser.add_argument(
        "--retarget-posture-weight",
        type=float,
        default=5e-2,
        help="Regularization weight toward previous joint solution.",
    )
    parser.add_argument(
        "--motion-smoothing",
        type=float,
        default=0.25,
        help="Exponential smoothing alpha for wrist pose targets (0-1).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    parser.add_argument("--perf", action="store_true", help="Log per-frame timing breakdown.")
    return parser.parse_args()


def _build_pre_step(
    latest: dict[str, HandFrame | HeadFrame],
    *,
    camera_name: str,
    max_iters: int,
    damping: float,
    step_size: float,
    tol: float,
    posture_weight: float,
    motion_smoothing: float,
) -> Any:
    state: dict[str, Any] = {}

    def pre_step(model: Any, data: Any) -> None:
        import mujoco

        if not state:
            # Initialize from home pose when available.
            try:
                key_id = model.key("home").id
                mujoco.mj_resetDataKeyframe(model, data, key_id)
            except Exception:
                pass
            mujoco.mj_forward(model, data)

            # Camera tracker: keep 6-DOF relative head motion behavior.
            cam_id = model.camera(camera_name).id
            flip_y_180 = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
            state["head_tracker"] = RelativeHeadCamera(
                model,
                cam_id,
                _INSPIRE_BASIS,
                track_position=True,
                head_rot_correction=flip_y_180,
            )

            side_state: dict[str, Any] = {}
            for side, cfg in _SIDE_CONFIG.items():
                base_ids = [model.actuator(name).id for name in cfg["base_actuators"]]
                home_pos = np.array([
                    data.ctrl[base_ids[0]],
                    data.ctrl[base_ids[1]],
                    data.ctrl[base_ids[2]],
                ])
                wrist_tracker = RelativeWristTracker(_INSPIRE_BASIS, home_pos, np.eye(3))

                finger_actuator_ids = []
                for jname in cfg["finger_joints"]:
                    aname = f"{jname.removesuffix('_joint')}_position"
                    finger_actuator_ids.append(model.actuator(aname).id)

                retargeter = MujocoVectorRetargeter(
                    model,
                    joint_names=cfg["finger_joints"],
                    site_by_joint=cfg["site_by_joint"],
                    tasks=default_tasks(),
                    basis=_INSPIRE_BASIS,
                    damping=damping,
                    step_size=step_size,
                    max_iters=max_iters,
                    tol=tol,
                    posture_weight=posture_weight,
                    auto_scale=True,
                )
                ball_qpos_adr = int(model.jnt_qposadr[model.joint(cfg["ball_joint"]).id])

                side_state[side] = {
                    "frame_key": cfg["frame_key"],
                    "base_ids": base_ids,
                    "wrist_tracker": wrist_tracker,
                    "finger_actuator_ids": np.array(finger_actuator_ids, dtype=np.int32),
                    "retargeter": retargeter,
                    "ball_qpos_adr": ball_qpos_adr,
                    "ball_dof_adr": int(model.jnt_dofadr[model.joint(cfg["ball_joint"]).id]),
                    "smoothed_pos": None,
                    "smoothed_rot": None,
                }

            state["sides"] = side_state

        # Head tracking.
        head = latest.get("Head")
        if isinstance(head, HeadFrame):
            state["head_tracker"].update(head, model)

        # Hand tracking + vector retargeting.
        for side, s in state["sides"].items():
            frame = latest.get(s["frame_key"])
            if not isinstance(frame, HandFrame):
                continue

            target_pos, target_rot = s["wrist_tracker"].update(frame.wrist)
            alpha = float(np.clip(motion_smoothing, 0.0, 1.0))
            if s["smoothed_pos"] is None:
                s["smoothed_pos"] = target_pos.copy()
                s["smoothed_rot"] = target_rot.copy()
            else:
                s["smoothed_pos"] = (1.0 - alpha) * s["smoothed_pos"] + alpha * target_pos
                # Matrix blend then re-orthonormalize with SVD.
                rot_blend = (1.0 - alpha) * s["smoothed_rot"] + alpha * target_rot
                u, _, vh = np.linalg.svd(rot_blend)
                s["smoothed_rot"] = u @ vh

            base_ids = s["base_ids"]
            data.ctrl[base_ids[0]] = s["smoothed_pos"][0]
            data.ctrl[base_ids[1]] = s["smoothed_pos"][1]
            data.ctrl[base_ids[2]] = s["smoothed_pos"][2]

            # Ball joint has no actuator in current Inspire XML, so set qpos directly.
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, s["smoothed_rot"].flatten())
            qadr = s["ball_qpos_adr"]
            data.qpos[qadr: qadr + 4] = quat
            # Avoid physics-injected wrist jitter from direct qpos writes.
            dadr = s["ball_dof_adr"]
            data.qvel[dadr: dadr + 3] = 0.0

            q_finger = s["retargeter"].solve(frame, full_qpos=data.qpos)
            data.ctrl[s["finger_actuator_ids"]] = q_finger

    return pre_step


async def _run() -> int:
    args = _parse_args()

    pre_step = None
    if not args.disable_mocap_tcp:
        latest = start_mocap_pump(args.mocap_tcp_host, args.mocap_tcp_port)
        pre_step = _build_pre_step(
            latest,
            camera_name=args.mj_camera,
            max_iters=args.retarget_iters,
            damping=args.retarget_damping,
            step_size=args.retarget_step,
            tol=args.retarget_tol,
            posture_weight=args.retarget_posture_weight,
            motion_smoothing=args.motion_smoothing,
        )

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


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
