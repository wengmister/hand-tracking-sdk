"""Run Shadow Hand E3M5 bimanual video host with vector retargeting.

Maps Quest hand tracking to two Shadow Hand E3M5 dexterous hands using
``MujocoVectorRetargeter`` for finger pose optimization.  Tendon-coupled
actuators (FFJ0, MFJ0, RFJ0, LFJ0) are driven by summing the solved
middle + distal joint positions.

Usage::

    uv run examples/video/shadow_hand_video_host.py --mocap-tcp-port 5555
    uv run examples/video/shadow_hand_video_host.py --preset 1080p --perf
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

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "assets", "shadow_hand", "scene_bimanual.xml"
)

# Palm-down home rotation: fingers +Y, palm -Z, thumb ±X.
_HOME_ROT = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

_SIDE_CONFIG: dict[str, dict[str, Any]] = {
    "left": {
        "frame_key": "Left",
        "base_actuators": (
            "lh_pos_x_position",
            "lh_pos_y_position",
            "lh_pos_z_position",
        ),
        "ball_joint": "lh_rot",
        "finger_joints": [
            "lh_THJ5",
            "lh_THJ4",
            "lh_THJ3",
            "lh_THJ2",
            "lh_THJ1",
            "lh_FFJ4",
            "lh_FFJ3",
            "lh_FFJ2",
            "lh_FFJ1",
            "lh_MFJ4",
            "lh_MFJ3",
            "lh_MFJ2",
            "lh_MFJ1",
            "lh_RFJ4",
            "lh_RFJ3",
            "lh_RFJ2",
            "lh_RFJ1",
            "lh_LFJ5",
            "lh_LFJ4",
            "lh_LFJ3",
            "lh_LFJ2",
            "lh_LFJ1",
        ],
        "site_by_joint": {
            JointName.WRIST: "lh_palm_site",
            JointName.THUMB_TIP: "lh_th_tip",
            JointName.INDEX_TIP: "lh_ff_tip",
            JointName.MIDDLE_TIP: "lh_mf_tip",
            JointName.RING_TIP: "lh_rf_tip",
            JointName.LITTLE_TIP: "lh_lf_tip",
        },
        "direct_joint_to_actuator": {
            "lh_THJ5": "lh_A_THJ5",
            "lh_THJ4": "lh_A_THJ4",
            "lh_THJ3": "lh_A_THJ3",
            "lh_THJ2": "lh_A_THJ2",
            "lh_THJ1": "lh_A_THJ1",
            "lh_FFJ4": "lh_A_FFJ4",
            "lh_FFJ3": "lh_A_FFJ3",
            "lh_MFJ4": "lh_A_MFJ4",
            "lh_MFJ3": "lh_A_MFJ3",
            "lh_RFJ4": "lh_A_RFJ4",
            "lh_RFJ3": "lh_A_RFJ3",
            "lh_LFJ5": "lh_A_LFJ5",
            "lh_LFJ4": "lh_A_LFJ4",
            "lh_LFJ3": "lh_A_LFJ3",
        },
        "tendon_actuator_joints": {
            "lh_A_FFJ0": ["lh_FFJ2", "lh_FFJ1"],
            "lh_A_MFJ0": ["lh_MFJ2", "lh_MFJ1"],
            "lh_A_RFJ0": ["lh_RFJ2", "lh_RFJ1"],
            "lh_A_LFJ0": ["lh_LFJ2", "lh_LFJ1"],
        },
    },
    "right": {
        "frame_key": "Right",
        "base_actuators": (
            "rh_pos_x_position",
            "rh_pos_y_position",
            "rh_pos_z_position",
        ),
        "ball_joint": "rh_rot",
        "finger_joints": [
            "rh_THJ5",
            "rh_THJ4",
            "rh_THJ3",
            "rh_THJ2",
            "rh_THJ1",
            "rh_FFJ4",
            "rh_FFJ3",
            "rh_FFJ2",
            "rh_FFJ1",
            "rh_MFJ4",
            "rh_MFJ3",
            "rh_MFJ2",
            "rh_MFJ1",
            "rh_RFJ4",
            "rh_RFJ3",
            "rh_RFJ2",
            "rh_RFJ1",
            "rh_LFJ5",
            "rh_LFJ4",
            "rh_LFJ3",
            "rh_LFJ2",
            "rh_LFJ1",
        ],
        "site_by_joint": {
            JointName.WRIST: "rh_palm_site",
            JointName.THUMB_TIP: "rh_th_tip",
            JointName.INDEX_TIP: "rh_ff_tip",
            JointName.MIDDLE_TIP: "rh_mf_tip",
            JointName.RING_TIP: "rh_rf_tip",
            JointName.LITTLE_TIP: "rh_lf_tip",
        },
        "direct_joint_to_actuator": {
            "rh_THJ5": "rh_A_THJ5",
            "rh_THJ4": "rh_A_THJ4",
            "rh_THJ3": "rh_A_THJ3",
            "rh_THJ2": "rh_A_THJ2",
            "rh_THJ1": "rh_A_THJ1",
            "rh_FFJ4": "rh_A_FFJ4",
            "rh_FFJ3": "rh_A_FFJ3",
            "rh_MFJ4": "rh_A_MFJ4",
            "rh_MFJ3": "rh_A_MFJ3",
            "rh_RFJ4": "rh_A_RFJ4",
            "rh_RFJ3": "rh_A_RFJ3",
            "rh_LFJ5": "rh_A_LFJ5",
            "rh_LFJ4": "rh_A_LFJ4",
            "rh_LFJ3": "rh_A_LFJ3",
        },
        "tendon_actuator_joints": {
            "rh_A_FFJ0": ["rh_FFJ2", "rh_FFJ1"],
            "rh_A_MFJ0": ["rh_MFJ2", "rh_MFJ1"],
            "rh_A_RFJ0": ["rh_RFJ2", "rh_RFJ1"],
            "rh_A_LFJ0": ["rh_LFJ2", "rh_LFJ1"],
        },
    },
}


def _parse_args() -> argparse.Namespace:
    parser = build_base_parser(
        "Host video service (Shadow Hand bimanual retarget).",
        mujoco=True,
        default_mj_model=_DEFAULT_MODEL,
        default_mj_camera="overview",
        default_preset="720p",
        default_mocap_port=5555,
    )
    parser.add_argument("--retarget-iters", type=int, default=5, help="GN iterations per frame.")
    parser.add_argument("--retarget-damping", type=float, default=1e-3, help="LM damping value.")
    parser.add_argument("--retarget-step", type=float, default=0.5, help="GN step size.")
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
            # Reset to home keyframe if available.
            try:
                key_id = model.key("home").id
                mujoco.mj_resetDataKeyframe(model, data, key_id)
            except Exception:
                pass
            mujoco.mj_forward(model, data)

            # Head camera tracker.
            cam_id = model.camera(camera_name).id
            state["head_tracker"] = RelativeHeadCamera(
                model,
                cam_id,
                position_transform=unity_left_to_rfu_position,
                rotation_matrix_transform=unity_left_to_rfu_rotation_matrix,
            )

            # Per-side initialization.
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
                    _HOME_ROT.copy(),
                    position_transform=unity_left_to_rfu_position,
                    rotation_matrix_transform=unity_left_to_rfu_rotation_matrix,
                )

                # Build retargeter for finger joints.
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

                # Precompute actuator index arrays for efficient writeback.
                joint_idx = {name: i for i, name in enumerate(cfg["finger_joints"])}

                direct_act_ids = []
                direct_q_idx = []
                for jname, aname in cfg["direct_joint_to_actuator"].items():
                    direct_act_ids.append(model.actuator(aname).id)
                    direct_q_idx.append(joint_idx[jname])

                tendon_act_ids = []
                tendon_q_idx_pairs = []
                for aname, jnames in cfg["tendon_actuator_joints"].items():
                    tendon_act_ids.append(model.actuator(aname).id)
                    tendon_q_idx_pairs.append([joint_idx[jn] for jn in jnames])

                ball_jid = model.joint(cfg["ball_joint"]).id
                side_state[side] = {
                    "frame_key": cfg["frame_key"],
                    "base_ids": base_ids,
                    "wrist_tracker": wrist_tracker,
                    "retargeter": retargeter,
                    "direct_act_ids": np.array(direct_act_ids, dtype=np.int32),
                    "direct_q_idx": np.array(direct_q_idx, dtype=np.int32),
                    "tendon_act_ids": tendon_act_ids,
                    "tendon_q_idx_pairs": tendon_q_idx_pairs,
                    "ball_qpos_adr": int(model.jnt_qposadr[ball_jid]),
                    "ball_dof_adr": int(model.jnt_dofadr[ball_jid]),
                    "smoothed_pos": None,
                    "smoothed_rot": None,
                }

            state["sides"] = side_state

        # Head tracking.
        head = latest.get("Head")
        if isinstance(head, HeadFrame):
            state["head_tracker"].update(head, model)

        # Hand tracking + vector retargeting per side.
        for _side, s in state["sides"].items():
            frame = latest.get(s["frame_key"])
            if not isinstance(frame, HandFrame):
                continue

            # Wrist pose (position via slide actuators, rotation via ball joint).
            target_pos, target_rot = s["wrist_tracker"].update(frame.wrist)

            alpha = float(np.clip(motion_smoothing, 0.0, 1.0))
            if s["smoothed_pos"] is None:
                s["smoothed_pos"] = target_pos.copy()
                s["smoothed_rot"] = target_rot.copy()
            else:
                s["smoothed_pos"] = (1.0 - alpha) * s["smoothed_pos"] + alpha * target_pos
                rot_blend = (1.0 - alpha) * s["smoothed_rot"] + alpha * target_rot
                u, _, vh = np.linalg.svd(rot_blend)
                s["smoothed_rot"] = u @ vh

            # Write position to slide actuators.
            base_ids = s["base_ids"]
            data.ctrl[base_ids[0]] = s["smoothed_pos"][0]
            data.ctrl[base_ids[1]] = s["smoothed_pos"][1]
            data.ctrl[base_ids[2]] = s["smoothed_pos"][2]

            # Write rotation to ball joint qpos (no actuator).
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, s["smoothed_rot"].flatten())
            qadr = s["ball_qpos_adr"]
            data.qpos[qadr : qadr + 4] = quat
            dadr = s["ball_dof_adr"]
            data.qvel[dadr : dadr + 3] = 0.0

            # Finger retargeting.
            q = s["retargeter"].solve(frame, full_qpos=data.qpos)

            # Write direct joint actuators.
            data.ctrl[s["direct_act_ids"]] = q[s["direct_q_idx"]]

            # Write tendon-coupled actuators (ctrl = sum of coupled joints).
            for act_id, q_indices in zip(s["tendon_act_ids"], s["tendon_q_idx_pairs"], strict=True):
                data.ctrl[act_id] = sum(q[i] for i in q_indices)

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
