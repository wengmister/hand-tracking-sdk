"""Run host-side MuJoCo video service for Unitree G1 with active vision.

Uses mink inverse kinematics to map incoming wrist poses to G1 arm
joint angles.  Head tracking data drives waist joints (yaw/roll/pitch)
to move a camera attached to the torso, giving the user an egocentric
first-person view.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
from typing import Any

from _common import compensate_gravity, run_video_service, start_mocap_pump

from hand_tracking_sdk.convert import (
    basis_transform_position,
    basis_transform_rotation_matrix,
)
from hand_tracking_sdk.frame import HandFrame, HeadFrame
from hand_tracking_sdk.video.service import VideoServiceConfig

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "assets", "unitree_g1", "scene_with_hands.xml"
)

# Basis matrix mapping Unity left-handed → G1 sim world frame (FLU).
# Unity left-handed: x=right, y=up, z=forward.
# G1 sim world (FLU): x=forward, y=left, z=up.
# Mapping: Unity_z → G1_x, -Unity_x → G1_y, Unity_y → G1_z.
_G1_BASIS = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)

# G1 arm joint names per side (7-DOF).
_ARM_JOINT_NAMES = [
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
]

# Waist joint names in kinematic chain order.
_WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

# Tighter soft limits for head-tracking-driven waist control.
# XML joint limits are ±0.52 rad (±30°) which topples the robot.
# These keep the torso COM inside the foot support polygon.
_WAIST_SOFT_LIMITS = {
    "waist_yaw_joint": (-0.3, 0.3),     # ±17°
    "waist_roll_joint": (-0.15, 0.15),   # ±8.6°
    "waist_pitch_joint": (-0.15, 0.15),  # ±8.6°
}

# Leg joint suffixes per side.
_LEG_JOINT_SUFFIXES = [
    "hip_pitch_joint",
    "hip_roll_joint",
    "hip_yaw_joint",
    "knee_joint",
    "ankle_pitch_joint",
    "ankle_roll_joint",
]

# Finger joint suffixes per side.
_FINGER_JOINT_SUFFIXES = [
    "hand_thumb_0_joint",
    "hand_thumb_1_joint",
    "hand_thumb_2_joint",
    "hand_middle_0_joint",
    "hand_middle_1_joint",
    "hand_index_0_joint",
    "hand_index_1_joint",
]


def _decompose_zxy_euler(rot: Any) -> tuple[float, float, float]:
    """Decompose 3x3 rotation matrix into Z-X-Y Euler angles.

    Returns ``(yaw, roll, pitch)`` matching the waist joint chain:
    ``waist_yaw(Z) -> waist_roll(X) -> waist_pitch(Y)``.

    For R = Rz(alpha) @ Rx(beta) @ Ry(gamma):
    - beta  = asin(R[2,1])
    - alpha = atan2(-R[0,1], R[1,1])
    - gamma = atan2(-R[2,0], R[2,2])
    """
    import numpy as np

    sin_beta = float(np.clip(rot[2, 1], -1.0, 1.0))
    beta = math.asin(sin_beta)
    alpha = math.atan2(-float(rot[0, 1]), float(rot[1, 1]))
    gamma = math.atan2(-float(rot[2, 0]), float(rot[2, 2]))
    return alpha, beta, gamma


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Host video service (G1 MuJoCo source)."
    )
    parser.add_argument(
        "--mj-model",
        default=_DEFAULT_MODEL,
        help="Path to MuJoCo XML model (default: bundled G1 with hands).",
    )
    parser.add_argument(
        "--mj-camera",
        default="head_cam",
        help="MuJoCo camera name (default: head_cam; use 'overview' to debug).",
    )
    parser.add_argument(
        "--tcp-host",
        default="0.0.0.0",
        help="WebSocket signaling bind host.",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=8765,
        help="WebSocket signaling bind port.",
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
        "--verbose",
        action="store_true",
        help="Enable verbose logs.",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Log per-frame timing breakdown.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# IK-based pre_step wiring
# ---------------------------------------------------------------------------


def _build_pre_step(
    latest: dict[str, HandFrame | HeadFrame],
) -> Any:
    """Build a pre_step callback for G1 arm teleop and active vision.

    Uses mink inverse kinematics for 7-DOF arm teleop and direct Euler
    angle mapping for waist-driven active vision from head tracking.
    """
    state: dict[str, Any] = {}

    def pre_step(model: Any, data: Any) -> None:
        import mujoco
        import numpy as np

        # ---- lazy initialization on first call ----
        if not state:
            try:
                import mink
            except ImportError as exc:
                print(f"[g1-host] mink not available: {exc}")
                return

            # Arm joint names.
            arm_joint_names: list[str] = []
            for prefix in ("left", "right"):
                for jn in _ARM_JOINT_NAMES:
                    arm_joint_names.append(f"{prefix}_{jn}")

            # Velocity limits for arm + waist joints only.
            velocity_limits: dict[str, float] = {}
            for name in arm_joint_names:
                velocity_limits[name] = np.pi
            for wn in _WAIST_JOINT_NAMES:
                velocity_limits[wn] = np.pi

            # Build a mask of velocity DOFs we actually want to move.
            # The G1 has a freejoint (6 DOFs) + legs + fingers that IK
            # should not touch.  We let mink solve freely, then zero
            # out everything except arm + waist before integrating.
            allowed_nv = set()
            for name in arm_joint_names:
                jid = model.joint(name).id
                nv_start = model.jnt_dofadr[jid]
                allowed_nv.add(nv_start)
            for wn in _WAIST_JOINT_NAMES:
                jid = model.joint(wn).id
                nv_start = model.jnt_dofadr[jid]
                allowed_nv.add(nv_start)
            vel_mask = np.zeros(model.nv, dtype=bool)
            for idx in allowed_nv:
                vel_mask[idx] = True

            # Use jnt_qposadr (not joint ID) to index into qpos.
            # The freejoint uses 7 qpos slots, shifting all subsequent
            # hinge joints' addresses by +6 from their joint IDs.
            arm_qpos_adr = np.array(
                [model.jnt_qposadr[model.joint(n).id] for n in arm_joint_names]
            )
            arm_actuator_ids = np.array(
                [model.actuator(n).id for n in arm_joint_names]
            )

            waist_actuator_ids = np.array(
                [model.actuator(n).id for n in _WAIST_JOINT_NAMES]
            )

            # Waist soft limits for clamping (tighter than XML joint limits).
            waist_limits = [
                _WAIST_SOFT_LIMITS[wn] for wn in _WAIST_JOINT_NAMES
            ]

            # Leg and finger actuator IDs (held at stand keyframe).
            leg_actuator_ids = np.array([
                model.actuator(f"{prefix}_{suffix}").id
                for prefix in ("left", "right")
                for suffix in _LEG_JOINT_SUFFIXES
            ])
            finger_actuator_ids = np.array([
                model.actuator(f"{prefix}_{suffix}").id
                for prefix in ("left", "right")
                for suffix in _FINGER_JOINT_SUFFIXES
            ])

            # Subtrees for gravity compensation (arm chains only).
            left_subtree = model.body("left_shoulder_pitch_link").id
            right_subtree = model.body("right_shoulder_pitch_link").id

            configuration = mink.Configuration(model)

            l_ee_task = mink.FrameTask(
                frame_name="left_ee",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )
            r_ee_task = mink.FrameTask(
                frame_name="right_ee",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )
            posture_task = mink.PostureTask(model, cost=1e-1)

            tasks = [l_ee_task, r_ee_task, posture_task]
            limits = [
                mink.ConfigurationLimit(model=model),
                mink.VelocityLimit(model, velocity_limits),
            ]

            # Reset to stand keyframe.
            mujoco.mj_resetDataKeyframe(
                model, data, model.key("stand").id
            )
            configuration.update(data.qpos)
            mujoco.mj_forward(model, data)
            posture_task.set_target_from_configuration(configuration)
            l_ee_task.set_target_from_configuration(configuration)
            r_ee_task.set_target_from_configuration(configuration)

            # Capture initial EE poses for differential teleop.
            l_site_id = model.site("left_ee").id
            r_site_id = model.site("right_ee").id

            # Capture stand keyframe ctrl for legs and fingers.
            stand_ctrl = data.ctrl.copy()

            initial_l_pos = data.site_xpos[l_site_id].copy()
            initial_r_pos = data.site_xpos[r_site_id].copy()
            print(
                f"[g1-host] stand EE poses: "
                f"left={initial_l_pos} right={initial_r_pos}"
            )

            state.update(
                mink=mink,
                configuration=configuration,
                vel_mask=vel_mask,
                l_ee_task=l_ee_task,
                r_ee_task=r_ee_task,
                tasks=tasks,
                limits=limits,
                arm_qpos_adr=arm_qpos_adr,
                arm_actuator_ids=arm_actuator_ids,
                waist_actuator_ids=waist_actuator_ids,
                waist_limits=waist_limits,
                leg_actuator_ids=leg_actuator_ids,
                finger_actuator_ids=finger_actuator_ids,
                subtree_ids=[left_subtree, right_subtree],
                initial_l_pos=initial_l_pos,
                initial_l_rot=data.site_xmat[l_site_id].reshape(3, 3).copy(),
                initial_r_pos=initial_r_pos,
                initial_r_rot=data.site_xmat[r_site_id].reshape(3, 3).copy(),
                ref_left_pos=None,
                ref_left_rot=None,
                ref_right_pos=None,
                ref_right_rot=None,
                ref_head_rot=None,
                stand_ctrl=stand_ctrl,
                _debug_count=0,
            )

        # ---- per-frame ----
        mink = state["mink"]
        configuration = state["configuration"]
        l_ee_task = state["l_ee_task"]
        r_ee_task = state["r_ee_task"]
        tasks = state["tasks"]
        limits = state["limits"]

        configuration.update(data.qpos)

        dt = 1.0 / 30.0

        # === Active Vision: head tracking → waist joints ===
        head = latest.get("Head")
        if isinstance(head, HeadFrame):
            h = head.head
            cur_rot = np.array(
                basis_transform_rotation_matrix(
                    h.qx, h.qy, h.qz, h.qw, _G1_BASIS
                )
            )
            if state["ref_head_rot"] is None:
                state["ref_head_rot"] = cur_rot.copy()

            delta_rot = cur_rot @ state["ref_head_rot"].T
            yaw, roll, pitch = _decompose_zxy_euler(delta_rot)

            waist_limits = state["waist_limits"]
            angles = [
                float(np.clip(yaw, waist_limits[0][0], waist_limits[0][1])),
                float(np.clip(roll, waist_limits[1][0], waist_limits[1][1])),
                float(np.clip(pitch, waist_limits[2][0], waist_limits[2][1])),
            ]
            data.ctrl[state["waist_actuator_ids"]] = angles

        # === Arm Teleop: hand tracking → IK ===
        left = latest.get("Left")
        right = latest.get("Right")
        state["_debug_count"] += 1
        _debug = state["_debug_count"] % 120 == 0  # log every ~2s

        if isinstance(left, HandFrame):
            w = left.wrist
            cur_pos = np.array(
                basis_transform_position((w.x, w.y, w.z), _G1_BASIS)
            )
            cur_rot = np.array(
                basis_transform_rotation_matrix(
                    w.qx, w.qy, w.qz, w.qw, _G1_BASIS
                )
            )
            if state["ref_left_pos"] is None:
                state["ref_left_pos"] = cur_pos.copy()
                state["ref_left_rot"] = cur_rot.copy()
            delta_pos = cur_pos - state["ref_left_pos"]
            delta_rot = cur_rot @ state["ref_left_rot"].T
            target_pos = state["initial_l_pos"] + delta_pos
            target_rot = delta_rot @ state["initial_l_rot"]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, target_rot.flatten())
            l_ee_task.set_target(
                mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=quat),
                    translation=target_pos,
                )
            )
            if _debug:
                print(f"[g1-host] L delta={delta_pos} target={target_pos}")

        if isinstance(right, HandFrame):
            w = right.wrist
            cur_pos = np.array(
                basis_transform_position((w.x, w.y, w.z), _G1_BASIS)
            )
            cur_rot = np.array(
                basis_transform_rotation_matrix(
                    w.qx, w.qy, w.qz, w.qw, _G1_BASIS
                )
            )
            if state["ref_right_pos"] is None:
                state["ref_right_pos"] = cur_pos.copy()
                state["ref_right_rot"] = cur_rot.copy()
            delta_pos = cur_pos - state["ref_right_pos"]
            delta_rot = cur_rot @ state["ref_right_rot"].T
            target_pos = state["initial_r_pos"] + delta_pos
            target_rot = delta_rot @ state["initial_r_rot"]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, target_rot.flatten())
            r_ee_task.set_target(
                mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=quat),
                    translation=target_pos,
                )
            )
            if _debug:
                print(f"[g1-host] R delta={delta_pos} target={target_pos}")

        # Solve IK (2 iterations for small per-frame deltas).
        # Mask velocity to only arm + waist DOFs so IK can't move
        # the freejoint, legs, or fingers.
        mask = state["vel_mask"]
        for _ in range(2):
            vel = mink.solve_ik(
                configuration,
                tasks,
                dt,
                "daqp",
                limits=limits,
                damping=1e-5,
            )
            vel[~mask] = 0.0
            configuration.integrate_inplace(vel, dt)

            l_err = l_ee_task.compute_error(configuration)
            r_err = r_ee_task.compute_error(configuration)
            if (
                np.linalg.norm(l_err[:3]) <= 5e-3
                and np.linalg.norm(l_err[3:]) <= 5e-3
                and np.linalg.norm(r_err[:3]) <= 5e-3
                and np.linalg.norm(r_err[3:]) <= 5e-3
            ):
                break

        # Write arm joint angles from IK solution.
        arm_qpos_adr = state["arm_qpos_adr"]
        arm_actuator_ids = state["arm_actuator_ids"]
        data.ctrl[arm_actuator_ids] = configuration.q[arm_qpos_adr]

        # Hold legs and fingers at stand keyframe.
        stand_ctrl = state["stand_ctrl"]
        data.ctrl[state["leg_actuator_ids"]] = stand_ctrl[
            state["leg_actuator_ids"]
        ]
        data.ctrl[state["finger_actuator_ids"]] = stand_ctrl[
            state["finger_actuator_ids"]
        ]

        # Gravity compensation on arm subtrees.
        compensate_gravity(model, data, state["subtree_ids"])

    return pre_step


def _build_perf_hook(interval: int = 60) -> Any:
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


async def _run() -> int:
    args = _parse_args()

    pre_step = None
    if not args.disable_mocap_tcp:
        latest = start_mocap_pump(args.mocap_tcp_host, args.mocap_tcp_port)
        pre_step = _build_pre_step(latest)

    perf_hook = _build_perf_hook() if args.perf else None

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
