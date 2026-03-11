"""Relative tracking helpers for MuJoCo-based video host scripts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hand_tracking_sdk.convert import Matrix3x3
from hand_tracking_sdk.frame import HeadFrame


class RelativeHeadCamera:
    """Track relative head motion and apply it to a MuJoCo camera.

    Records the first head frame as reference, then applies position and
    rotation deltas to the camera's XML defaults on each update.
    """

    def __init__(
        self,
        model: Any,
        cam_id: int,
        basis: Matrix3x3 | None = None,
        *,
        position_transform: (
            Callable[[float, float, float], tuple[float, float, float]] | None
        ) = None,
        rotation_matrix_transform: Callable[
            [float, float, float, float], Matrix3x3
        ]
        | None = None,
        track_position: bool = False,
        head_rot_correction: Any | None = None,
    ) -> None:
        import mujoco
        import numpy as np

        self._cam_id = cam_id
        self._basis = basis
        self._position_transform = position_transform
        self._rotation_matrix_transform = rotation_matrix_transform
        self._track_position = track_position
        self._head_rot_correction = head_rot_correction

        self._default_pos = model.cam_pos[cam_id].copy()
        default_mat = np.zeros(9)
        mujoco.mju_quat2Mat(default_mat, model.cam_quat[cam_id])
        self._default_rot = default_mat.reshape(3, 3).copy()

        self._ref_pos: Any | None = None
        self._ref_rot: Any | None = None

    def update(self, head_frame: HeadFrame, model: Any) -> None:
        """Apply relative head delta to camera transform."""
        import mujoco
        import numpy as np

        from hand_tracking_sdk.convert import (
            basis_transform_position,
            basis_transform_rotation_matrix,
        )

        h = head_frame.head
        if self._rotation_matrix_transform is not None:
            head_rot = np.asarray(
                self._rotation_matrix_transform(h.qx, h.qy, h.qz, h.qw), dtype=float
            )
        elif self._basis is not None:
            head_rot = np.asarray(
                basis_transform_rotation_matrix(h.qx, h.qy, h.qz, h.qw, self._basis), dtype=float
            )
        else:
            raise ValueError("RelativeHeadCamera requires basis or rotation_matrix_transform.")
        if self._head_rot_correction is not None:
            head_rot = head_rot @ self._head_rot_correction

        if self._ref_rot is None:
            self._ref_rot = head_rot.copy()
            if self._track_position:
                if self._position_transform is not None:
                    head_pos = np.asarray(self._position_transform(h.x, h.y, h.z), dtype=float)
                elif self._basis is not None:
                    head_pos = np.asarray(
                        basis_transform_position((h.x, h.y, h.z), self._basis), dtype=float
                    )
                else:
                    raise ValueError("RelativeHeadCamera requires basis or position_transform.")
                self._ref_pos = head_pos.copy()
            return

        assert self._ref_rot is not None
        delta_rot = head_rot @ self._ref_rot.T
        cam_rot = delta_rot @ self._default_rot
        cam_quat = np.empty(4)
        mujoco.mju_mat2Quat(cam_quat, cam_rot.flatten())
        model.cam_quat[self._cam_id] = cam_quat

        if self._track_position:
            if self._position_transform is not None:
                head_pos = np.asarray(self._position_transform(h.x, h.y, h.z), dtype=float)
            elif self._basis is not None:
                head_pos = np.asarray(
                    basis_transform_position((h.x, h.y, h.z), self._basis), dtype=float
                )
            else:
                raise ValueError("RelativeHeadCamera requires basis or position_transform.")
            model.cam_pos[self._cam_id] = self._default_pos + (head_pos - self._ref_pos)


class RelativeWristTracker:
    """Track relative wrist motion from a reference frame.

    Returns target position and rotation by applying deltas from the
    first observed wrist pose to given home/initial transforms.
    """

    def __init__(
        self,
        basis: Matrix3x3 | None,
        home_pos: Any,
        home_rot: Any,
        *,
        position_transform: (
            Callable[[float, float, float], tuple[float, float, float]] | None
        ) = None,
        rotation_matrix_transform: Callable[
            [float, float, float, float], Matrix3x3
        ]
        | None = None,
    ) -> None:
        self._basis = basis
        self._position_transform = position_transform
        self._rotation_matrix_transform = rotation_matrix_transform
        self._home_pos = home_pos
        self._home_rot = home_rot
        self._ref_pos: Any | None = None
        self._ref_rot: Any | None = None

    def update(self, wrist: Any) -> tuple[Any, Any]:
        """Compute target position and rotation from current wrist pose.

        Returns ``(target_pos, target_rot)`` as numpy arrays.
        """
        import numpy as np

        from hand_tracking_sdk.convert import (
            basis_transform_position,
            basis_transform_rotation_matrix,
        )

        if self._position_transform is not None:
            cur_pos = np.asarray(self._position_transform(wrist.x, wrist.y, wrist.z), dtype=float)
        elif self._basis is not None:
            cur_pos = np.asarray(
                basis_transform_position((wrist.x, wrist.y, wrist.z), self._basis), dtype=float
            )
        else:
            raise ValueError("RelativeWristTracker requires basis or position_transform.")

        if self._rotation_matrix_transform is not None:
            cur_rot = np.asarray(
                self._rotation_matrix_transform(wrist.qx, wrist.qy, wrist.qz, wrist.qw), dtype=float
            )
        elif self._basis is not None:
            cur_rot = np.asarray(
                basis_transform_rotation_matrix(
                    wrist.qx, wrist.qy, wrist.qz, wrist.qw, self._basis
                ),
                dtype=float,
            )
        else:
            raise ValueError("RelativeWristTracker requires basis or rotation_matrix_transform.")
        if self._ref_pos is None:
            self._ref_pos = cur_pos.copy()
            self._ref_rot = cur_rot.copy()

        delta_pos = cur_pos - self._ref_pos
        assert self._ref_rot is not None
        delta_rot = cur_rot @ self._ref_rot.T
        target_pos = self._home_pos + delta_pos
        target_rot = delta_rot @ self._home_rot
        return target_pos, target_rot
