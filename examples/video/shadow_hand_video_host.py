"""Run Shadow Hand (right) host with vector retargeting.

Maps Quest hand tracking data to the Shadow Hand E3M5 dexterous hand model
using the MujocoVectorRetargeter.  The hand is fixed in space (no wrist
translation); wrist flex/extend and abduction are driven via the model's
own wrist hinge joints.  Four tendon-coupled actuators (FFJ0, MFJ0, RFJ0,
LFJ0) are driven by summing the solved middle + distal joint positions.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

import numpy as np
from _common import build_perf_hook, run_video_service, start_mocap_pump
from _retarget import MujocoVectorRetargeter, default_tasks
from _tracking import RelativeHeadCamera

from hand_tracking_sdk.frame import HandFrame, HeadFrame
from hand_tracking_sdk.models import JointName
from hand_tracking_sdk.video.service import VideoServiceConfig

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "assets", "shadow_hand", "scene_retarget.xml"
)

# Unity left-handed (x right, y up, z forward) -> MuJoCo (x right, y forward, z up).
_SHADOW_BASIS = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
)

# All hinge joints to optimize (right hand).
_FINGER_JOINTS = [
    "rh_WRJ2",
    "rh_WRJ1",
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
]

# Site mapping for retargeter vector tasks.
_SITE_MAP: dict[JointName, str] = {
    JointName.WRIST: "rh_palm_site",
    JointName.THUMB_TIP: "rh_th_tip",
    JointName.INDEX_TIP: "rh_ff_tip",
    JointName.MIDDLE_TIP: "rh_mf_tip",
    JointName.RING_TIP: "rh_rf_tip",
    JointName.LITTLE_TIP: "rh_lf_tip",
}

# Joints with a 1:1 position actuator.
_DIRECT_JOINT_TO_ACTUATOR: dict[str, str] = {
    "rh_WRJ2": "rh_A_WRJ2",
    "rh_WRJ1": "rh_A_WRJ1",
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
}

# Tendon actuators: ctrl = sum of coupled joint qpos values.
_TENDON_ACTUATOR_JOINTS: dict[str, list[str]] = {
    "rh_A_FFJ0": ["rh_FFJ2", "rh_FFJ1"],
    "rh_A_MFJ0": ["rh_MFJ2", "rh_MFJ1"],
    "rh_A_RFJ0": ["rh_RFJ2", "rh_RFJ1"],
    "rh_A_LFJ0": ["rh_LFJ2", "rh_LFJ1"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host video service (Shadow Hand retarget).")
    parser.add_argument(
        "--mj-model",
        default=_DEFAULT_MODEL,
        help="Path to MuJoCo XML model (default: bundled Shadow Hand scene).",
    )
    parser.add_argument(
        "--mj-camera", default="overview", help="MuJoCo camera name or id string."
    )
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
    parser.add_argument(
        "--retarget-iters", type=int, default=5, help="GN iterations per frame."
    )
    parser.add_argument(
        "--retarget-damping", type=float, default=1e-3, help="LM damping value."
    )
    parser.add_argument("--retarget-step", type=float, default=0.5, help="GN step size.")
    parser.add_argument("--retarget-tol", type=float, default=1e-4, help="Early-stop dq norm.")
    parser.add_argument(
        "--retarget-posture-weight",
        type=float,
        default=5e-2,
        help="Regularization weight toward previous joint solution.",
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
) -> Any:
    state: dict[str, Any] = {}

    def pre_step(model: Any, data: Any) -> None:
        import mujoco

        if not state:
            # Reset to "open hand" keyframe if available.
            try:
                key_id = model.key("open hand").id
                mujoco.mj_resetDataKeyframe(model, data, key_id)
            except Exception:
                pass
            mujoco.mj_forward(model, data)

            # Head camera tracker.
            cam_id = model.camera(camera_name).id
            state["head_tracker"] = RelativeHeadCamera(
                model, cam_id, _SHADOW_BASIS,
            )

            # Build retargeter.
            retargeter = MujocoVectorRetargeter(
                model,
                joint_names=_FINGER_JOINTS,
                site_by_joint=_SITE_MAP,
                tasks=default_tasks(),
                basis=_SHADOW_BASIS,
                damping=damping,
                step_size=step_size,
                max_iters=max_iters,
                tol=tol,
                posture_weight=posture_weight,
                auto_scale=True,
            )

            # Precompute actuator index arrays for efficient writeback.
            joint_idx = {name: i for i, name in enumerate(_FINGER_JOINTS)}

            direct_act_ids = []
            direct_q_idx = []
            for jname, aname in _DIRECT_JOINT_TO_ACTUATOR.items():
                direct_act_ids.append(model.actuator(aname).id)
                direct_q_idx.append(joint_idx[jname])

            tendon_act_ids = []
            tendon_q_idx_pairs = []
            for aname, jnames in _TENDON_ACTUATOR_JOINTS.items():
                tendon_act_ids.append(model.actuator(aname).id)
                tendon_q_idx_pairs.append([joint_idx[jn] for jn in jnames])

            state.update(
                retargeter=retargeter,
                direct_act_ids=np.array(direct_act_ids, dtype=np.int32),
                direct_q_idx=np.array(direct_q_idx, dtype=np.int32),
                tendon_act_ids=tendon_act_ids,
                tendon_q_idx_pairs=tendon_q_idx_pairs,
            )

        # Head tracking.
        head = latest.get("Head")
        if isinstance(head, HeadFrame):
            state["head_tracker"].update(head, model)

        # Right hand retargeting.
        frame = latest.get("Right")
        if not isinstance(frame, HandFrame):
            return

        q = state["retargeter"].solve(frame, full_qpos=data.qpos)

        # Write direct joint actuators.
        data.ctrl[state["direct_act_ids"]] = q[state["direct_q_idx"]]

        # Write tendon-coupled actuators (ctrl = sum of coupled joints).
        for act_id, q_indices in zip(
            state["tendon_act_ids"], state["tendon_q_idx_pairs"], strict=True
        ):
            data.ctrl[act_id] = sum(q[i] for i in q_indices)

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
