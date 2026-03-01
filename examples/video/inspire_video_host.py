"""Run host-side Inspire bimanual hand video service with mocap-driven teleop.

Maps Quest hand tracking data to the Inspire dexterous hand model using
relative wrist positioning and finger curl retargeting.  All tracking
(hands + head) is relative: the first frame from the Quest is recorded as
a reference and subsequent frames apply deltas to sensible home positions
defined in the scene keyframe.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
from typing import Any

from _common import run_video_service, start_mocap_pump

from hand_tracking_sdk.convert import basis_transform_position, basis_transform_rotation_matrix
from hand_tracking_sdk.frame import HandFrame, HeadFrame
from hand_tracking_sdk.models import JointName
from hand_tracking_sdk.teleop import finger_curl_angles
from hand_tracking_sdk.video.service import VideoServiceConfig

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "assets", "inspire", "scene_bimanual.xml")

# Basis matrix mapping Unity left-handed → MuJoCo world frame.
# Unity left-handed: x=right, y=up, z=forward.
# MuJoCo world:      x=right, y=forward, z=up.
# Mapping: (x,y,z) → (x, z, y).
_INSPIRE_BASIS = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
)

# Base actuator suffixes (position then rotation).
_BASE_SUFFIXES = ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")

# Finger joint suffixes per finger, ordered to match curl angle output.
# Thumb's first entry (yaw) is handled via _thumb_yaw(), rest from curl angles.
# SDK uses "little" but Inspire model uses "pinky" — the mapping here bridges that.
_FINGER_JOINTS: dict[str, list[str]] = {
    "thumb": [
        "thumb_proximal_yaw",
        "thumb_proximal_pitch",
        "thumb_intermediate",
        "thumb_distal",
    ],
    "index": ["index_proximal", "index_intermediate"],
    "middle": ["middle_proximal", "middle_intermediate"],
    "ring": ["ring_proximal", "ring_intermediate"],
    "little": ["pinky_proximal", "pinky_intermediate"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host video service (Inspire hands).")
    parser.add_argument(
        "--mj-model",
        default=_DEFAULT_MODEL,
        help="Path to MuJoCo XML model (default: bundled Inspire bimanual).",
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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    parser.add_argument("--perf", action="store_true", help="Log per-frame timing breakdown.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _decompose_xyz_euler(rot: Any) -> tuple[float, float, float]:
    """Decompose a 3x3 rotation matrix into intrinsic XYZ Euler angles.

    For the joint chain rot_x → rot_y → rot_z, the combined rotation is
    R = Rx(α) @ Ry(β) @ Rz(γ).

    Returns (α, β, γ) in radians.
    """
    # R[0,2] = sin(β)
    sy = float(rot[0, 2])
    sy = max(-1.0, min(1.0, sy))
    beta = math.asin(sy)

    if abs(sy) < 0.9999:
        # Non-gimbal-lock case.
        alpha = math.atan2(-float(rot[1, 2]), float(rot[2, 2]))
        gamma = math.atan2(-float(rot[0, 1]), float(rot[0, 0]))
    else:
        # Gimbal lock: set γ=0, solve α from remaining.
        gamma = 0.0
        alpha = math.atan2(float(rot[1, 0]), float(rot[1, 1]))

    return (alpha, beta, gamma)


def _vec_sub(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _angle_between(
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> float:
    """Angle in radians between two 3-D vectors."""
    dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    len_u = math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    len_v = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    denom = len_u * len_v
    if denom == 0.0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, dot / denom)))


def _thumb_yaw(frame: HandFrame, ctrl_range: tuple[float, float]) -> float:
    """Compute thumb abduction angle from spread between thumb and index.

    Maps the spread angle linearly to the given actuator control range.
    """
    wrist = frame.get_joint(JointName.WRIST)
    thumb_mcp = frame.get_joint(JointName.THUMB_METACARPAL)
    index_prox = frame.get_joint(JointName.INDEX_PROXIMAL)

    v_thumb = _vec_sub(thumb_mcp, wrist)
    v_index = _vec_sub(index_prox, wrist)
    spread = _angle_between(v_thumb, v_index)

    # Calibration: thumb tucked ≈ 0.3 rad, fully spread ≈ 1.2 rad.
    lo, hi = ctrl_range
    t = (spread - 0.3) / (1.2 - 0.3)
    t = max(0.0, min(1.0, t))
    return lo + t * (hi - lo)


def _scale_curl(angle: float, joint_range: tuple[float, float]) -> float:
    """Scale a curl angle (0 to π) to a joint range."""
    lo, hi = joint_range
    t = angle / math.pi
    t = max(0.0, min(1.0, t))
    return lo + t * (hi - lo)


# ---------------------------------------------------------------------------
# pre_step callback
# ---------------------------------------------------------------------------


def _build_pre_step(
    latest: dict[str, HandFrame | HeadFrame],
    *,
    camera_name: str = "overview",
) -> Any:
    """Build a pre_step callback with fully relative tracking.

    All tracking (hands + head) is relative: the first Quest frame is
    recorded as a reference and subsequent frames apply deltas to the
    home positions defined in the scene keyframe.
    """
    import numpy as np

    # Mutable state populated on first call (lazy init).
    state: dict[str, Any] = {}

    def pre_step(model: Any, data: Any) -> None:
        import mujoco

        # ---- lazy initialization on first call ----
        if not state:
            # Reset to "home" keyframe so hands start at sensible positions.
            key_id = model.key(0).id
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            mujoco.mj_forward(model, data)

            # Resolve actuator IDs and control ranges from the model.
            base_ids: dict[str, list[int]] = {}
            finger_info: dict[str, list[tuple[int, float, float]]] = {}
            initial_pos: dict[str, Any] = {}

            for side in ("left", "right"):
                ids = [
                    model.actuator(f"{side}_{s}_position").id for s in _BASE_SUFFIXES
                ]
                base_ids[side] = ids

                # Read home position from keyframe ctrl values.
                initial_pos[side] = np.array([
                    data.ctrl[ids[0]], data.ctrl[ids[1]], data.ctrl[ids[2]],
                ])

                info: list[tuple[int, float, float]] = []
                for joints in _FINGER_JOINTS.values():
                    for jname in joints:
                        aid = model.actuator(f"{side}_{jname}_position").id
                        lo, hi = model.actuator_ctrlrange[aid]
                        info.append((aid, float(lo), float(hi)))
                finger_info[side] = info

            cam_id = model.camera(camera_name).id

            # Save XML camera defaults for relative head tracking.
            cam_default_pos = model.cam_pos[cam_id].copy()
            cam_default_mat = np.zeros(9)
            mujoco.mju_quat2Mat(cam_default_mat, model.cam_quat[cam_id])
            cam_default_rot = cam_default_mat.reshape(3, 3).copy()

            state.update(
                base_ids=base_ids,
                finger_info=finger_info,
                initial_pos=initial_pos,
                cam_id=cam_id,
                cam_default_pos=cam_default_pos,
                cam_default_rot=cam_default_rot,
            )

        base_ids = state["base_ids"]
        finger_info = state["finger_info"]

        side_map = {"left": "Left", "right": "Right"}

        for side, frame_key in side_map.items():
            frame = latest.get(frame_key)
            if not isinstance(frame, HandFrame):
                continue

            w = frame.wrist
            ids = base_ids[side]

            cur_pos = np.array(
                basis_transform_position((w.x, w.y, w.z), _INSPIRE_BASIS)
            )
            cur_rot = np.array(
                basis_transform_rotation_matrix(w.qx, w.qy, w.qz, w.qw, _INSPIRE_BASIS)
            )

            # --- Relative wrist tracking ---
            ref_key = f"ref_{side}"
            if ref_key not in state:
                # First frame for this hand: record as reference.
                state[ref_key] = (cur_pos.copy(), cur_rot.copy())

            ref_pos, ref_rot = state[ref_key]
            home_pos = state["initial_pos"][side]

            # Position: home + delta from first Quest frame.
            target_pos = home_pos + (cur_pos - ref_pos)
            data.ctrl[ids[0]] = target_pos[0]
            data.ctrl[ids[1]] = target_pos[1]
            data.ctrl[ids[2]] = target_pos[2]

            # Rotation: the palm-down orientation is baked into the orient body
            # quat in the XML, so the hinge joints only represent the rotation
            # delta from the user's initial hand pose.  Near identity the XYZ
            # Euler decomposition is well-conditioned (β ≈ 0, far from ±π/2).
            delta_rot = cur_rot @ ref_rot.T
            alpha, beta, gamma = _decompose_xyz_euler(delta_rot)
            data.ctrl[ids[3]] = alpha
            data.ctrl[ids[4]] = beta
            data.ctrl[ids[5]] = gamma

            # --- Finger retargeting (scale-independent, no change needed) ---
            curls = finger_curl_angles(frame)
            finfo = finger_info[side]
            idx = 0

            # Thumb (4 DOF): yaw from spread, then 3 curl angles.
            aid, lo, hi = finfo[idx]
            data.ctrl[aid] = _thumb_yaw(frame, (lo, hi))
            idx += 1
            for ci in range(3):
                aid, lo, hi = finfo[idx]
                data.ctrl[aid] = _scale_curl(curls["thumb"][ci], (lo, hi))
                idx += 1

            # Remaining fingers (2 DOF each): proximal + intermediate curls.
            for finger in ("index", "middle", "ring", "little"):
                for ci in range(2):
                    aid, lo, hi = finfo[idx]
                    data.ctrl[aid] = _scale_curl(curls[finger][ci], (lo, hi))
                    idx += 1

        # --- 6DOF head-tracked camera (relative to initial head pose) ---
        head = latest.get("Head")
        if isinstance(head, HeadFrame):
            h = head.head
            cam_id = state["cam_id"]

            head_pos = np.array(
                basis_transform_position((h.x, h.y, h.z), _INSPIRE_BASIS)
            )
            head_rot = np.array(
                basis_transform_rotation_matrix(h.qx, h.qy, h.qz, h.qw, _INSPIRE_BASIS)
            )
            # MuJoCo camera looks along -Z; Quest head looks along +Z.
            flip_y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
            head_rot = head_rot @ flip_y_180

            if "head_ref_pos" not in state:
                state["head_ref_pos"] = head_pos.copy()
                state["head_ref_rot"] = head_rot.copy()
            else:
                ref_pos = state["head_ref_pos"]
                ref_rot = state["head_ref_rot"]
                default_pos = state["cam_default_pos"]
                default_rot = state["cam_default_rot"]

                model.cam_pos[cam_id] = default_pos + (head_pos - ref_pos)

                delta_rot = head_rot @ ref_rot.T
                cam_rot = delta_rot @ default_rot

                cam_quat = np.empty(4)
                mujoco.mju_mat2Quat(cam_quat, cam_rot.flatten())
                model.cam_quat[cam_id] = cam_quat

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
        pre_step = _build_pre_step(
            latest,
            camera_name=args.mj_camera,
        )

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
