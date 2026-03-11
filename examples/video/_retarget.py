"""
Lightweight vector-based hand retargeting for MuJoCo.

This module implements a small, dependency-light optimizer that matches a set
of task vectors (for example wrist->fingertip and thumb->index) between a
tracked human hand and a robot hand in MuJoCo.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from hand_tracking_sdk.constants import STREAMED_JOINT_NAMES
from hand_tracking_sdk.convert import Matrix3x3, basis_transform_rotation_matrix
from hand_tracking_sdk.frame import HandFrame
from hand_tracking_sdk.models import HandLandmarks, HandSide, JointName, WristPose

PointKey = str | JointName


def _point_key_name(key: PointKey) -> str:
    return key.value if isinstance(key, JointName) else key


@dataclass(frozen=True, slots=True)
class VectorTask:
    """One vector-matching objective.

    The solver tries to match:
        robot(point_b - point_a) ~= (scale * human(joint_b - joint_a))
    """

    point_a: PointKey
    point_b: PointKey
    weight: float = 1.0
    scale: float = 1.0


def default_tasks(
    *,
    wrist: PointKey = JointName.WRIST,
    thumb_tip: PointKey = JointName.THUMB_TIP,
    index_tip: PointKey = JointName.INDEX_TIP,
    middle_tip: PointKey = JointName.MIDDLE_TIP,
    ring_tip: PointKey = JointName.RING_TIP,
    little_tip: PointKey = JointName.LITTLE_TIP,
) -> list[VectorTask]:
    """Return a practical default task set for dexterous retargeting."""
    wrist_to_tips = [
        VectorTask(wrist, thumb_tip, weight=1.0),
        VectorTask(wrist, index_tip, weight=1.0),
        VectorTask(wrist, middle_tip, weight=1.0),
        VectorTask(wrist, ring_tip, weight=0.8),
        VectorTask(wrist, little_tip, weight=0.8),
    ]
    pinch = [
        VectorTask(thumb_tip, index_tip, weight=1.3),
        VectorTask(thumb_tip, middle_tip, weight=1.0),
    ]
    return wrist_to_tips + pinch


@dataclass(frozen=True, slots=True)
class RobotHandSpec:
    """Reusable mapping spec for one robot hand."""

    joint_names: list[str]
    site_by_point: dict[PointKey, str]
    tasks: list[VectorTask] | None = None


_IDENTITY_BASIS: Matrix3x3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class MujocoVectorRetargeter:
    """Vector-based hand retargeting solver for one robot hand."""

    def __init__(
        self,
        model: Any,
        *,
        joint_names: list[str] | None = None,
        site_by_joint: Mapping[PointKey, str] | None = None,
        hand_spec: RobotHandSpec | None = None,
        tasks: list[VectorTask] | None = None,
        position_transform: (
            Callable[[float, float, float], tuple[float, float, float]] | None
        ) = None,
        rotation_matrix_transform: (
            Callable[
                [float, float, float, float],
                Matrix3x3,
            ]
            | None
        ) = None,
        damping: float = 1e-4,
        step_size: float = 1.0,
        max_iters: int = 8,
        tol: float = 1e-4,
        posture_weight: float = 1e-3,
        auto_scale: bool = True,
        landmarks_wrist_relative: bool = True,
        point_extractor: Callable[[HandFrame, PointKey], tuple[float, float, float]] | None = None,
    ) -> None:
        """Create retargeter.

        Parameters:
        - model: MuJoCo model
        - joint_names: robot joint names to optimize (single hand chain)
        - site_by_joint: mapping from point label -> MuJoCo site name
        - hand_spec: optional convenience bundle for joint/site/task mapping
        - tasks: vector tasks; defaults to :func:`default_tasks`
        - position_transform: explicit position transform callback
        - rotation_matrix_transform: explicit quaternion->rotation callback
        - damping: Levenberg-Marquardt damping
        - step_size: global step scale for each GN update
        - max_iters: iterations per solve call
        - tol: early-stop threshold on ||dq||
        - posture_weight: keeps solution close to previous q
        - auto_scale: calibrate a global human->robot scale on first frame
        - landmarks_wrist_relative: if True, rotate landmark vectors by wrist
          orientation before matching robot world-space vectors
        - point_extractor: optional callback to fetch one source point from
          HandFrame for non-standard label sets
        """
        self.model = model
        import mujoco

        self.data = mujoco.MjData(model)

        if hand_spec is not None:
            if joint_names is None:
                joint_names = hand_spec.joint_names
            if site_by_joint is None:
                site_by_joint = hand_spec.site_by_point
            if tasks is None:
                tasks = hand_spec.tasks
        if joint_names is None or site_by_joint is None:
            raise ValueError("Provide (joint_names + site_by_joint) or hand_spec.")

        self.tasks = tasks or default_tasks()
        self._position_transform = position_transform
        self._rotation_matrix_transform = rotation_matrix_transform
        self.damping = float(damping)
        self.step_size = float(step_size)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.posture_weight = float(posture_weight)
        self.auto_scale = bool(auto_scale)
        self.landmarks_wrist_relative = bool(landmarks_wrist_relative)
        self._point_extractor = point_extractor

        self._site_ids: dict[str, int] = {
            _point_key_name(p): model.site(site_name).id for p, site_name in site_by_joint.items()
        }
        for task in self.tasks:
            a = _point_key_name(task.point_a)
            b = _point_key_name(task.point_b)
            if a not in self._site_ids or b not in self._site_ids:
                raise ValueError(
                    f"Missing site mapping for task {_point_key_name(task.point_a)}"
                    f"->{_point_key_name(task.point_b)}"
                )

        self._joint_qpos: npt.NDArray[Any]
        self._joint_dof: npt.NDArray[Any]
        qpos_ids: list[int] = []
        dof_ids: list[int] = []
        lower: list[float] = []
        upper: list[float] = []

        import mujoco

        for name in joint_names:
            jid = model.joint(name).id
            jtype = int(model.jnt_type[jid])
            if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(f"Joint {name!r} is not hinge/slide; got joint type {jtype}")
            qid = int(model.jnt_qposadr[jid])
            did = int(model.jnt_dofadr[jid])
            qpos_ids.append(qid)
            dof_ids.append(did)
            lo, hi = model.jnt_range[jid]
            lower.append(float(lo))
            upper.append(float(hi))

        self._joint_qpos = np.array(qpos_ids, dtype=np.int32)
        self._joint_dof = np.array(dof_ids, dtype=np.int32)
        self._lower = np.array(lower, dtype=float)
        self._upper = np.array(upper, dtype=float)

        self._prev_q: npt.NDArray[Any] | None = None
        self._global_scale: float | None = None

    def reset_reference(self) -> None:
        """Reset solver memory (posture target and auto-scale calibration)."""
        self._prev_q = None
        self._global_scale = None

    def solve(
        self,
        frame: HandFrame,
        *,
        q_init: npt.NDArray[Any] | None = None,
        full_qpos: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Solve for optimized joint positions for this hand.

        Returns a vector aligned with ``joint_names`` passed at construction.
        """
        import mujoco

        target_vectors = self._target_vectors(frame)
        if self._global_scale is None and self.auto_scale:
            self._global_scale = self._estimate_scale(target_vectors, full_qpos=full_qpos)
        global_scale = 1.0 if self._global_scale is None else self._global_scale

        if full_qpos is None:
            self.data.qpos[:] = 0.0
        else:
            self.data.qpos[:] = full_qpos

        if q_init is None:
            if self._prev_q is not None:
                q = self._prev_q.copy()
            else:
                q = np.clip(self.data.qpos[self._joint_qpos], self._lower, self._upper)
        else:
            q = np.clip(q_init.astype(float, copy=True), self._lower, self._upper)

        q_ref = q.copy() if self._prev_q is None else self._prev_q

        # Gauss-Newton on position-vector residuals.
        for _ in range(self.max_iters):
            self.data.qpos[self._joint_qpos] = q
            mujoco.mj_forward(self.model, self.data)

            residuals, jac_rows = self._linearized_system(
                target_vectors=target_vectors,
                global_scale=global_scale,
            )
            if self.posture_weight > 0.0:
                w = np.sqrt(self.posture_weight)
                residuals = np.concatenate([residuals, w * (q - q_ref)])
                jac_rows.append(w * np.eye(q.shape[0], dtype=float))

            j = np.vstack(jac_rows)
            # Normal equations with LM damping.
            jt = j.T
            h = jt @ j + self.damping * np.eye(j.shape[1], dtype=float)
            g = jt @ residuals
            try:
                dq = -np.linalg.solve(h, g)
            except np.linalg.LinAlgError:
                dq = -np.linalg.pinv(h) @ g

            if self.step_size != 1.0:
                dq *= self.step_size

            q = np.clip(q + dq, self._lower, self._upper)
            if float(np.linalg.norm(dq)) <= self.tol:
                break

        self._prev_q = q.copy()
        return q

    def residual_norm(
        self,
        frame: HandFrame,
        q: npt.NDArray[Any],
        *,
        full_qpos: npt.NDArray[Any] | None = None,
    ) -> float:
        """Compute weighted vector residual norm at a given joint configuration."""
        import mujoco

        if full_qpos is None:
            self.data.qpos[:] = 0.0
        else:
            self.data.qpos[:] = full_qpos
        self.data.qpos[self._joint_qpos] = np.clip(q, self._lower, self._upper)
        mujoco.mj_forward(self.model, self.data)

        target_vectors = self._target_vectors(frame)
        if self._global_scale is None and self.auto_scale:
            scale = self._estimate_scale(target_vectors, full_qpos=full_qpos)
        else:
            scale = 1.0 if self._global_scale is None else self._global_scale

        residual_chunks: list[npt.NDArray[Any]] = []
        for task, target in zip(self.tasks, target_vectors, strict=False):
            a_id = self._site_ids[_point_key_name(task.point_a)]
            b_id = self._site_ids[_point_key_name(task.point_b)]
            robot_vec = self.data.site_xpos[b_id] - self.data.site_xpos[a_id]
            desired = scale * target
            w = np.sqrt(float(task.weight))
            residual_chunks.append(w * (robot_vec - desired))
        residual = np.concatenate(residual_chunks)
        return float(np.linalg.norm(residual))

    def _target_vectors(self, frame: HandFrame) -> list[npt.NDArray[Any]]:
        vectors: list[npt.NDArray[Any]] = []
        for task in self.tasks:
            a = self._frame_joint(frame, task.point_a)
            b = self._frame_joint(frame, task.point_b)
            vec = (b - a) * float(task.scale)
            vectors.append(vec)
        return vectors

    def _raw_frame_point(self, frame: HandFrame, key: PointKey) -> tuple[float, float, float]:
        if self._point_extractor is not None:
            return self._point_extractor(frame, key)
        return frame.get_joint(key)

    def _frame_joint(self, frame: HandFrame, name: PointKey) -> npt.NDArray[Any]:
        p = self._raw_frame_point(frame, name)
        if self.landmarks_wrist_relative:
            # HTS landmarks are wrist-relative vectors. Transform coordinates, then rotate
            # by wrist orientation so the target vectors are in world frame.
            if self._position_transform is not None:
                p_conv = np.array(self._position_transform(p[0], p[1], p[2]), dtype=float)
            else:
                p_conv = np.array(p, dtype=float)
            w = frame.wrist
            if self._rotation_matrix_transform is not None:
                wrist_rot = np.array(
                    self._rotation_matrix_transform(w.qx, w.qy, w.qz, w.qw),
                    dtype=float,
                )
            else:
                wrist_rot = np.array(
                    basis_transform_rotation_matrix(w.qx, w.qy, w.qz, w.qw, _IDENTITY_BASIS),
                    dtype=float,
                )
            result: npt.NDArray[Any] = wrist_rot @ p_conv
            return result

        if self._position_transform is not None:
            return np.array(self._position_transform(p[0], p[1], p[2]), dtype=float)
        return np.array(p, dtype=float)

    def _estimate_scale(
        self,
        target_vectors: list[npt.NDArray[Any]],
        *,
        full_qpos: npt.NDArray[Any] | None,
    ) -> float:
        import mujoco

        if full_qpos is None:
            self.data.qpos[:] = 0.0
        else:
            self.data.qpos[:] = full_qpos
        mujoco.mj_forward(self.model, self.data)

        robot_norms: list[float] = []
        human_norms: list[float] = []
        for task, hvec in zip(self.tasks, target_vectors, strict=False):
            a_id = self._site_ids[_point_key_name(task.point_a)]
            b_id = self._site_ids[_point_key_name(task.point_b)]
            rvec = self.data.site_xpos[b_id] - self.data.site_xpos[a_id]
            rn = float(np.linalg.norm(rvec))
            hn = float(np.linalg.norm(hvec))
            if rn > 1e-8 and hn > 1e-8:
                robot_norms.append(rn)
                human_norms.append(hn)
        if not robot_norms:
            return 1.0
        return float(np.mean(robot_norms) / np.mean(human_norms))

    def _linearized_system(
        self,
        *,
        target_vectors: list[npt.NDArray[Any]],
        global_scale: float,
    ) -> tuple[npt.NDArray[Any], list[npt.NDArray[Any]]]:
        import mujoco

        residual_chunks: list[npt.NDArray[Any]] = []
        jac_rows: list[npt.NDArray[Any]] = []

        jac_pos_a = np.zeros((3, self.model.nv), dtype=float)
        jac_pos_b = np.zeros((3, self.model.nv), dtype=float)

        for task, target in zip(self.tasks, target_vectors, strict=False):
            a_id = self._site_ids[_point_key_name(task.point_a)]
            b_id = self._site_ids[_point_key_name(task.point_b)]

            pa = self.data.site_xpos[a_id]
            pb = self.data.site_xpos[b_id]
            robot_vec = pb - pa
            desired = global_scale * target

            w = np.sqrt(float(task.weight))
            residual_chunks.append(w * (robot_vec - desired))

            mujoco.mj_jacSite(self.model, self.data, jac_pos_a, None, a_id)
            mujoco.mj_jacSite(self.model, self.data, jac_pos_b, None, b_id)
            jv = jac_pos_b[:, self._joint_dof] - jac_pos_a[:, self._joint_dof]
            jac_rows.append(w * jv)

        return np.concatenate(residual_chunks), jac_rows


_INSPIRE_SIDE_SITES: dict[str, dict[PointKey, str]] = {
    "left": {
        JointName.WRIST: "left_palm",
        JointName.THUMB_TIP: "left_thumb_tip",
        JointName.INDEX_TIP: "left_index_tip",
        JointName.MIDDLE_TIP: "left_middle_tip",
        JointName.RING_TIP: "left_ring_tip",
        JointName.LITTLE_TIP: "left_pinky_tip",
    },
    "right": {
        JointName.WRIST: "right_palm",
        JointName.THUMB_TIP: "right_thumb_tip",
        JointName.INDEX_TIP: "right_index_tip",
        JointName.MIDDLE_TIP: "right_middle_tip",
        JointName.RING_TIP: "right_ring_tip",
        JointName.LITTLE_TIP: "right_pinky_tip",
    },
}

_INSPIRE_SIDE_JOINTS: dict[str, list[str]] = {
    "left": [
        "left_pos_x",
        "left_pos_y",
        "left_pos_z",
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
    "right": [
        "right_pos_x",
        "right_pos_y",
        "right_pos_z",
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
}


def inspire_hand_spec(side: str) -> RobotHandSpec:
    """Return a reusable Inspire hand mapping spec for ``left`` or ``right``."""
    if side not in _INSPIRE_SIDE_JOINTS:
        raise ValueError(f"Unknown side {side!r}; expected 'left' or 'right'.")
    return RobotHandSpec(
        joint_names=_INSPIRE_SIDE_JOINTS[side],
        site_by_point=_INSPIRE_SIDE_SITES[side],
        tasks=default_tasks(),
    )


def _default_inspire_scene() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "assets",
        "inspire",
        "scene_bimanual.xml",
    )


def _joint_index_by_name() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(STREAMED_JOINT_NAMES)}


def _build_frame_from_joint_points(
    side: HandSide,
    points: dict[PointKey, npt.NDArray[Any]],
) -> HandFrame:
    idx = _joint_index_by_name()
    arr = np.zeros((len(STREAMED_JOINT_NAMES), 3), dtype=float)
    for point_key, pos in points.items():
        joint_name = _point_key_name(point_key)
        if joint_name not in idx:
            continue
        arr[idx[joint_name]] = np.asarray(pos, dtype=float)
    wrist = arr[idx[JointName.WRIST.value]]
    landmarks = HandLandmarks(points=tuple((float(p[0]), float(p[1]), float(p[2])) for p in arr))
    return HandFrame(
        side=side,
        frame_id="retarget-smoke",
        wrist=WristPose(
            x=float(wrist[0]),
            y=float(wrist[1]),
            z=float(wrist[2]),
            qx=0.0,
            qy=0.0,
            qz=0.0,
            qw=1.0,
        ),
        landmarks=landmarks,
        sequence_id=0,
        recv_ts_ns=0,
        recv_time_unix_ns=None,
        source_ts_ns=None,
        wrist_recv_ts_ns=0,
        landmarks_recv_ts_ns=0,
    )


def _run_smoke_test(
    model_path: str,
    side: str,
    seed: int,
    noise_std: float,
    target_perturb: float,
) -> int:
    import mujoco

    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    home_qpos = np.zeros(model.nq, dtype=float)
    try:
        key_id = model.key("home").id
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        home_qpos[:] = data.qpos
    except (KeyError, ValueError):
        home_qpos[:] = data.qpos

    spec = inspire_hand_spec(side)
    joint_names = spec.joint_names
    sites = spec.site_by_point

    retargeter = MujocoVectorRetargeter(
        model,
        hand_spec=spec,
        auto_scale=False,
        max_iters=10,
    )

    # Build a synthetic "human" target by sampling a reachable robot pose.
    q_true = np.clip(
        home_qpos[retargeter._joint_qpos]
        + rng.uniform(-target_perturb, target_perturb, size=retargeter._joint_qpos.shape[0]),
        retargeter._lower,
        retargeter._upper,
    )
    data.qpos[:] = home_qpos
    data.qpos[retargeter._joint_qpos] = q_true
    mujoco.mj_forward(model, data)

    joint_points: dict[PointKey, npt.NDArray[Any]] = {}
    for jn, sidename in sites.items():
        sid = model.site(sidename).id
        joint_points[jn] = data.site_xpos[sid].copy()

    frame = _build_frame_from_joint_points(
        HandSide.LEFT if side == "left" else HandSide.RIGHT,
        joint_points,
    )

    q_init = np.clip(
        q_true + rng.normal(0.0, noise_std, size=q_true.shape[0]),
        retargeter._lower,
        retargeter._upper,
    )

    before_res = retargeter.residual_norm(frame, q_init, full_qpos=home_qpos)
    solved = retargeter.solve(frame, q_init=q_init, full_qpos=home_qpos)
    after_res = retargeter.residual_norm(frame, solved, full_qpos=home_qpos)
    rmse_before = float(np.sqrt(np.mean((q_init - q_true) ** 2)))
    rmse_after = float(np.sqrt(np.mean((solved - q_true) ** 2)))

    print(f"[retarget-smoke] model={model_path}")
    print(f"[retarget-smoke] side={side} joints={len(joint_names)} tasks={len(retargeter.tasks)}")
    print(f"[retarget-smoke] residual norm before={before_res:.6f} after={after_res:.6f}")
    print(
        "[retarget-smoke] q rmse to synthetic target "
        f"before={rmse_before:.6f} after={rmse_after:.6f}"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vector retargeting smoke test.")
    parser.add_argument(
        "--model",
        default=_default_inspire_scene(),
        help="MuJoCo XML path (default: Inspire bimanual scene).",
    )
    parser.add_argument(
        "--side",
        default="left",
        choices=("left", "right"),
        help="Which hand preset mapping to test.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.20,
        help="Initial joint noise std-dev (radians/meters).",
    )
    parser.add_argument(
        "--target-perturb",
        type=float,
        default=0.25,
        help="Synthetic target perturbation around home.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(
        _run_smoke_test(
            model_path=args.model,
            side=args.side,
            seed=args.seed,
            noise_std=args.noise_std,
            target_perturb=args.target_perturb,
        )
    )
