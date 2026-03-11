"""Run Inspire Hand bimanual video host with vector retargeting.

Maps Quest hand tracking to two Inspire dexterous hands using
``MujocoVectorRetargeter`` for finger pose optimization.

Usage::

    uv run examples/video/inspire_hand_video_host.py --mocap-tcp-port 5555
    uv run examples/video/inspire_hand_video_host.py --preset 1080p --perf
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

import numpy as np
from _common import build_base_parser, run_mujoco_host
from _retarget import MujocoVectorRetargeter, default_tasks
from _tracking import RelativeHeadCamera, RelativeWristTracker

from hand_tracking_sdk.convert import (
    unity_left_to_rfu_position,
    unity_left_to_rfu_rotation_matrix,
)
from hand_tracking_sdk.frame import HandFrame, HeadFrame
from hand_tracking_sdk.models import JointName

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "assets", "inspire", "scene_bimanual.xml")

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
    parser = build_base_parser(
        "Host video service (Inspire vector retarget).",
        mujoco=True,
        default_mj_model=_DEFAULT_MODEL,
        default_mj_camera="overview",
        default_preset="720p",
        default_mocap_port=5555,
    )
    parser.add_argument("--retarget-iters", type=int, default=5, help="GN iterations per frame.")
    parser.add_argument("--retarget-damping", type=float, default=1e-3, help="LM damping value.")
    parser.add_argument("--retarget-step", type=float, default=0.75, help="GN step size.")
    parser.add_argument("--retarget-tol", type=float, default=1e-4, help="Early-stop dq norm.")
    parser.add_argument(
        "--retarget-posture-weight",
        type=float,
        default=1e-2,
        help="Regularization weight toward previous joint solution.",
    )
    parser.add_argument(
        "--motion-smoothing",
        type=float,
        default=0.75,
        help="Exponential smoothing alpha for wrist pose targets (0-1).",
    )
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
            # 180° rotation about Y so the camera's forward axis aligns with the
            # Inspire hand's forward direction after coordinate conversion.
            flip_y_180 = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
            state["head_tracker"] = RelativeHeadCamera(
                model,
                cam_id,
                position_transform=unity_left_to_rfu_position,
                rotation_matrix_transform=unity_left_to_rfu_rotation_matrix,
                track_position=True,
                head_rot_correction=flip_y_180,
            )

            side_state: dict[str, Any] = {}
            for side, cfg in _SIDE_CONFIG.items():
                base_ids = [model.actuator(name).id for name in cfg["base_actuators"]]
                home_pos = np.array(
                    [
                        data.ctrl[base_ids[0]],
                        data.ctrl[base_ids[1]],
                        data.ctrl[base_ids[2]],
                    ]
                )
                wrist_tracker = RelativeWristTracker(
                    None,
                    home_pos,
                    np.eye(3),
                    position_transform=unity_left_to_rfu_position,
                    rotation_matrix_transform=unity_left_to_rfu_rotation_matrix,
                )

                finger_actuator_ids = []
                for jname in cfg["finger_joints"]:
                    aname = f"{jname.removesuffix('_joint')}_position"
                    finger_actuator_ids.append(model.actuator(aname).id)

                retargeter = MujocoVectorRetargeter(
                    model,
                    joint_names=cfg["finger_joints"],
                    site_by_joint=cfg["site_by_joint"],
                    tasks=default_tasks(),
                    position_transform=unity_left_to_rfu_position,
                    rotation_matrix_transform=unity_left_to_rfu_rotation_matrix,
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
        for _side, s in state["sides"].items():
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
            data.qpos[qadr : qadr + 4] = quat
            # Avoid physics-injected wrist jitter from direct qpos writes.
            dadr = s["ball_dof_adr"]
            data.qvel[dadr : dadr + 3] = 0.0

            q_finger = s["retargeter"].solve(frame, full_qpos=data.qpos)
            data.ctrl[s["finger_actuator_ids"]] = q_finger

    return pre_step


def _build_pre_step_from_args(
    latest: dict[str, HandFrame | HeadFrame],
    args: argparse.Namespace,
) -> Any:
    return _build_pre_step(
        latest,
        camera_name=args.mj_camera,
        max_iters=args.retarget_iters,
        damping=args.retarget_damping,
        step_size=args.retarget_step,
        tol=args.retarget_tol,
        posture_weight=args.retarget_posture_weight,
        motion_smoothing=args.motion_smoothing,
    )


async def _run() -> int:
    args = _parse_args()
    return await run_mujoco_host(args, _build_pre_step_from_args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
