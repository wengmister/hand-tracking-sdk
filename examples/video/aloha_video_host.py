"""Run ALOHA 2 bimanual video host with IK-based teleop.

Uses mink inverse kinematics to map incoming wrist poses to ALOHA joint
angles.  Gripper control is derived from thumb-to-index pinch distance
via the SDK teleop module.  Requires ``mink`` and ``daqp``.

Usage::

    uv run examples/video/aloha_video_host.py --mocap-tcp-port 5555
    uv run examples/video/aloha_video_host.py --preset 1080p --perf
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from _common import build_base_parser, compensate_gravity, run_mujoco_host
from _tracking import RelativeHeadCamera, RelativeWristTracker

from hand_tracking_sdk.convert import (
    unity_left_to_rfu_position,
    unity_left_to_rfu_rotation_matrix,
)
from hand_tracking_sdk.frame import HandFrame, HeadFrame
from hand_tracking_sdk.teleop import GripConfig, grip_value

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "assets", "aloha", "scene.xml")

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
    parser = build_base_parser(
        "Host video service (MuJoCo source).",
        mujoco=True,
        default_mj_model=_DEFAULT_MODEL,
        default_mj_camera="teleop_overview",
        default_preset="480p",
        default_mocap_port=5555,
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# IK-based pre_step wiring
# ---------------------------------------------------------------------------


def _build_pre_step(
    latest: dict[str, HandFrame | HeadFrame],
    *,
    left_gripper_actuator: str,
    right_gripper_actuator: str,
    camera_name: str = "teleop_overview",
    grip_config: GripConfig | None = None,
) -> Any:
    """Build a pre_step callback that applies mocap state to MuJoCo via IK.

    Uses mink inverse kinematics to map wrist poses from incoming hand
    frames to joint-position actuator commands for the ALOHA arms.
    Head tracking drives 3-DOF camera rotation on *camera_name*.
    Gripper control is derived from pinch distance.
    """
    if grip_config is None:
        grip_config = GripConfig()

    # Mutable state populated on first call (lazy init).
    state: dict[str, Any] = {}

    def pre_step(model: Any, data: Any) -> None:
        import mujoco
        import numpy as np

        # If initialization previously failed, do nothing.
        if state.get("disabled"):
            return

        # ---- lazy initialization on first call ----
        if not state:
            try:
                import mink
            except ImportError as exc:
                print(f"[mujoco-host] mink not available: {exc}")
                state["disabled"] = True
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

            head_cam = RelativeHeadCamera(
                model,
                model.camera(camera_name).id,
                position_transform=unity_left_to_rfu_position,
                rotation_matrix_transform=unity_left_to_rfu_rotation_matrix,
            )

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
                head_cam=head_cam,
                left_tracker=RelativeWristTracker(
                    None,
                    data.site_xpos[l_site_id].copy(),
                    data.site_xmat[l_site_id].reshape(3, 3).copy(),
                    position_transform=unity_left_to_rfu_position,
                    rotation_matrix_transform=unity_left_to_rfu_rotation_matrix,
                ),
                right_tracker=RelativeWristTracker(
                    None,
                    data.site_xpos[r_site_id].copy(),
                    data.site_xmat[r_site_id].reshape(3, 3).copy(),
                    position_transform=unity_left_to_rfu_position,
                    rotation_matrix_transform=unity_left_to_rfu_rotation_matrix,
                ),
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

        # === Head tracking → camera rotation ===
        head = latest.get("Head")
        if isinstance(head, HeadFrame):
            state["head_cam"].update(head, model)

        left = latest.get("Left")
        right = latest.get("Right")

        # Differential teleop: compute hand delta from reference pose,
        # apply to initial EE pose, then solve IK.
        if isinstance(left, HandFrame):
            target_pos, target_rot = state["left_tracker"].update(left.wrist)
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
            target_pos, target_rot = state["right_tracker"].update(right.wrist)
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
        compensate_gravity(model, data, state["subtree_ids"])

    return pre_step


def _build_pre_step_from_args(
    latest: dict[str, HandFrame | HeadFrame],
    args: argparse.Namespace,
) -> Any:
    return _build_pre_step(
        latest,
        left_gripper_actuator=args.left_gripper_actuator,
        right_gripper_actuator=args.right_gripper_actuator,
        camera_name=args.mj_camera,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_mujoco_host(_parse_args(), _build_pre_step_from_args)))
