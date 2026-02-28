"""Run host-side MuJoCo video service with optional mocap-driven teleop.

Uses mink inverse kinematics to map incoming wrist poses to ALOHA joint
angles.  Gripper control is derived from thumb-to-index pinch distance
via the SDK teleop module.
"""

from __future__ import annotations

import argparse
import asyncio
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
from hand_tracking_sdk.frame import HandFrame, HeadFrame
from hand_tracking_sdk.teleop import GripConfig, grip_value
from hand_tracking_sdk.video.service import VideoServiceConfig

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "assets", "aloha", "scene.xml")


# ---------------------------------------------------------------------------
# Unity left-handed → ALOHA sim world frame coordinate helpers
# ---------------------------------------------------------------------------


def _wrist_to_sim_pos(wrist: Any) -> Any:
    """Convert Unity left-handed wrist position to ALOHA sim world frame.

    Unity left-handed: x=right, y=up, z=forward.
    ALOHA sim world:   x=right (screen), y=forward (into screen), z=up.
    Mapping: (x,y,z) → (x, z, y).
    """
    import numpy as np

    return np.array([wrist.x, wrist.z, wrist.y])


def _wrist_to_sim_rotmat(wrist: Any) -> Any:
    """Convert Unity left-handed wrist quaternion to ALOHA sim 3x3 rotation matrix.

    WristPose stores quaternion as (qx, qy, qz, qw) — xyzw order.
    The basis transform maps Unity-left vectors to the ALOHA sim world
    frame.  Both arms use the same world-frame mapping — IK handles the
    per-arm kinematics.
    Rotation matrices transform as: R_sim = T @ R_unity @ T^T.
    """
    import mujoco
    import numpy as np

    basis = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
    rot = np.empty(9)
    # mju_quat2Mat expects wxyz order.
    mujoco.mju_quat2Mat(rot, [wrist.qw, wrist.qx, wrist.qy, wrist.qz])
    return basis @ rot.reshape(3, 3) @ basis.T


# ALOHA joint names per arm (6-DOF, no gripper).
_ARM_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host video service (MuJoCo source).")
    parser.add_argument(
        "--mj-model",
        default=_DEFAULT_MODEL,
        help="Path to MuJoCo XML model (default: bundled ALOHA).",
    )
    parser.add_argument(
        "--mj-camera", default="teleop_overview", help="MuJoCo camera name or id string."
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
        default="480p60",
        choices=("480p30", "480p60", "720p30", "720p60", "1080p30"),
        help="Video preset (default 480p60 for responsive teleop).",
    )
    parser.add_argument(
        "--left-gripper-actuator",
        default="left/gripper",
        help="MuJoCo actuator name for left gripper.",
    )
    parser.add_argument(
        "--right-gripper-actuator",
        default="right/gripper",
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
            latest[event.side.value] = event

    thread = Thread(target=_pump, daemon=True)
    thread.start()
    return latest


# ---------------------------------------------------------------------------
# Gravity compensation (from mink ALOHA example)
# ---------------------------------------------------------------------------


def _compensate_gravity(
    model: Any,
    data: Any,
    subtree_ids: list[int],
) -> None:
    """Apply gravity compensation forces to the given subtrees."""
    import mujoco
    import numpy as np

    data.qfrc_applied[:] = 0.0
    jac = np.empty((3, model.nv))
    for sid in subtree_ids:
        total_mass = model.body_subtreemass[sid]
        mujoco.mj_jacSubtreeCom(model, data, jac, sid)
        data.qfrc_applied[:] -= model.opt.gravity * total_mass @ jac


# ---------------------------------------------------------------------------
# IK-based pre_step wiring
# ---------------------------------------------------------------------------


def _build_pre_step(
    latest: dict[str, HandFrame | HeadFrame],
    *,
    left_gripper_actuator: str,
    right_gripper_actuator: str,
    grip_config: GripConfig | None = None,
) -> Any:
    """Build a pre_step callback that applies mocap state to MuJoCo via IK.

    Uses mink inverse kinematics to map wrist poses from incoming hand
    frames to joint-position actuator commands for the ALOHA arms.
    Gripper control is derived from pinch distance.
    """
    if grip_config is None:
        grip_config = GripConfig()

    # Mutable state populated on first call (lazy init).
    state: dict[str, Any] = {}

    def pre_step(model: Any, data: Any) -> None:
        import mujoco
        import numpy as np

        # ---- lazy initialization on first call ----
        if not state:
            try:
                import mink
            except ImportError as exc:
                print(f"[mujoco-host] mink not available: {exc}")
                return

            # Build joint name lists and resolve IDs.
            joint_names: list[str] = []
            velocity_limits: dict[str, float] = {}
            for prefix in ("left", "right"):
                for jn in _ARM_JOINT_NAMES:
                    name = f"{prefix}/{jn}"
                    joint_names.append(name)
                    velocity_limits[name] = np.pi

            dof_ids = np.array([model.joint(n).id for n in joint_names])
            actuator_ids = np.array([model.actuator(n).id for n in joint_names])

            left_grip_id = model.actuator(left_gripper_actuator).id
            right_grip_id = model.actuator(right_gripper_actuator).id

            left_subtree = model.body("left/base_link").id
            right_subtree = model.body("right/base_link").id

            configuration = mink.Configuration(model)

            l_ee_task = mink.FrameTask(
                frame_name="left/gripper",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )
            r_ee_task = mink.FrameTask(
                frame_name="right/gripper",
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

            # Reset to neutral pose and set initial targets.
            mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
            configuration.update(data.qpos)
            mujoco.mj_forward(model, data)
            posture_task.set_target_from_configuration(configuration)
            l_ee_task.set_target_from_configuration(configuration)
            r_ee_task.set_target_from_configuration(configuration)

            # Capture initial EE poses for differential teleop.
            l_site_id = model.site("left/gripper").id
            r_site_id = model.site("right/gripper").id

            state.update(
                mink=mink,
                configuration=configuration,
                l_ee_task=l_ee_task,
                r_ee_task=r_ee_task,
                tasks=tasks,
                limits=limits,
                dof_ids=dof_ids,
                actuator_ids=actuator_ids,
                left_grip_id=left_grip_id,
                right_grip_id=right_grip_id,
                subtree_ids=[left_subtree, right_subtree],
                initial_l_pos=data.site_xpos[l_site_id].copy(),
                initial_l_rot=data.site_xmat[l_site_id].reshape(3, 3).copy(),
                initial_r_pos=data.site_xpos[r_site_id].copy(),
                initial_r_rot=data.site_xmat[r_site_id].reshape(3, 3).copy(),
                ref_left_pos=None,
                ref_left_rot=None,
                ref_right_pos=None,
                ref_right_rot=None,
            )

        # ---- per-frame IK solve ----
        mink = state["mink"]
        configuration = state["configuration"]
        l_ee_task = state["l_ee_task"]
        r_ee_task = state["r_ee_task"]
        tasks = state["tasks"]
        limits = state["limits"]
        dof_ids = state["dof_ids"]
        actuator_ids = state["actuator_ids"]

        # Sync mink configuration with actual sim state after previous mj_step.
        configuration.update(data.qpos)

        dt = 1.0 / 30.0  # Match video frame rate.

        left = latest.get("Left")
        right = latest.get("Right")

        # Differential teleop: compute hand delta from reference pose,
        # apply to initial EE pose.  Positions and rotations are converted
        # from Unity left-handed to FLU before computing deltas so they
        # align with the MuJoCo world frame.
        if isinstance(left, HandFrame):
            cur_pos = _wrist_to_sim_pos(left.wrist)
            cur_rot = _wrist_to_sim_rotmat(left.wrist)
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
            data.ctrl[state["left_grip_id"]] = grip_value(left, grip_config)

        if isinstance(right, HandFrame):
            cur_pos = _wrist_to_sim_pos(right.wrist)
            cur_rot = _wrist_to_sim_rotmat(right.wrist)
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
            data.ctrl[state["right_grip_id"]] = grip_value(right, grip_config)

        # Solve IK (2 iterations suffices for small per-frame deltas).
        for _ in range(2):
            vel = mink.solve_ik(
                configuration,
                tasks,
                dt,
                "daqp",
                limits=limits,
                damping=1e-5,
            )
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

        # Write joint angles and apply gravity compensation.
        data.ctrl[actuator_ids] = configuration.q[dof_ids]
        _compensate_gravity(model, data, state["subtree_ids"])

    return pre_step


async def _run() -> int:
    args = _parse_args()

    # Set up mocap ingestion and pre_step hook when mocap is enabled.
    pre_step = None
    if not args.disable_mocap_tcp:
        latest = _start_mocap_pump(args.mocap_tcp_host, args.mocap_tcp_port)
        pre_step = _build_pre_step(
            latest,
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
