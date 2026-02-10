"""Optional real-time visualization helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType
from typing import cast

from hand_tracking_sdk.convert import (
    convert_hand_frame_unity_left_to_right,
    convert_landmarks_unity_left_to_right,
    convert_wrist_pose_unity_left_to_right,
)
from hand_tracking_sdk.frame import HandFrame
from hand_tracking_sdk.models import HandSide, LandmarksPacket, ParsedPacket, WristPacket, WristPose

from .exceptions import VisualizationDependencyError


class VisualizationFrame(StrEnum):
    """Output frame convention used for visualization points."""

    SDK = "sdk"
    FLU = "flu"


@dataclass(frozen=True, slots=True)
class RerunVisualizerConfig:
    """Configuration for :class:`RerunVisualizer`.

    :param application_id:
        Application identifier displayed in Rerun.
    :param spawn:
        If ``True``, spawn a local Rerun viewer on initialization.
    :param landmarks_are_wrist_relative:
        If ``True``, landmark points are interpreted as wrist-local coordinates
        and transformed into world frame before logging.
    :param wrist_radius:
        Point radius for wrist markers in meters.
    :param landmark_radius:
        Point radius for landmark markers in meters.
    :param wrist_color:
        RGB color for wrist markers.
    :param left_landmark_color:
        RGB color for left-hand landmark markers.
    :param right_landmark_color:
        RGB color for right-hand landmark markers.
    :param convert_to_right_handed:
        If ``True``, incoming Unity left-handed data are converted to right-handed
        coordinates before visualization.
    :param background_color:
        Optional RGB background color for the Rerun 3D view.
    :param visualization_frame:
        Point frame convention for visualization output.
        ``flu`` means ``x=forward, y=left, z=up``.
    """

    application_id: str = "hand-tracking-sdk"
    spawn: bool = True
    landmarks_are_wrist_relative: bool = True
    wrist_radius: float = 0.025
    landmark_radius: float = 0.015
    wrist_color: tuple[int, int, int] = (255, 220, 0)
    left_landmark_color: tuple[int, int, int] = (64, 128, 255)
    right_landmark_color: tuple[int, int, int] = (255, 64, 64)
    convert_to_right_handed: bool = True
    background_color: tuple[int, int, int] | None = (18, 22, 30)
    visualization_frame: VisualizationFrame = VisualizationFrame.FLU


class RerunVisualizer:
    """Visualizer that logs hand telemetry to `rerun`.

    This component is optional and requires installing the visualization extra.
    """

    def __init__(self, config: RerunVisualizerConfig | None = None) -> None:
        """Create a Rerun visualizer.

        :param config:
            Optional visualizer configuration.
        :raises VisualizationDependencyError:
            If `rerun-sdk` is not installed.
        """
        self._config = config or RerunVisualizerConfig()
        self._rr = self._import_rerun()
        self._latest_wrist_by_side: dict[HandSide, WristPose] = {}
        self._rr.init(self._config.application_id, spawn=self._config.spawn)
        self._apply_view_background()

    def log_packet(self, packet: ParsedPacket) -> None:
        """Log one parsed packet to Rerun.

        :param packet:
            Parsed packet event from HTS stream.
        """
        side_path = f"hands/{packet.side.value.lower()}"
        if isinstance(packet, WristPacket):
            pose = packet.data
            if self._config.convert_to_right_handed:
                pose = convert_wrist_pose_unity_left_to_right(pose)
            self._latest_wrist_by_side[packet.side] = pose
            self._log_points(
                f"{side_path}/wrist",
                [(pose.x, pose.y, pose.z)],
                radius=self._config.wrist_radius,
                color=self._config.wrist_color,
            )
            return

        if isinstance(packet, LandmarksPacket):
            points = packet.data.points
            if self._config.convert_to_right_handed:
                points = convert_landmarks_unity_left_to_right(packet.data).points
            if self._config.landmarks_are_wrist_relative:
                wrist = self._latest_wrist_by_side.get(packet.side)
                if wrist is None:
                    return
                points = self._transform_landmarks_by_wrist(points=points, wrist=wrist)
            self._log_points(
                f"{side_path}/landmarks",
                points,
                radius=self._config.landmark_radius,
                color=self._landmark_color(packet.side),
            )

    def log_frame(self, frame: HandFrame) -> None:
        """Log one assembled frame to Rerun.

        :param frame:
            Assembled hand frame.
        """
        visual_frame = (
            convert_hand_frame_unity_left_to_right(frame)
            if self._config.convert_to_right_handed
            else frame
        )
        base = f"frames/{visual_frame.frame_id}"
        self._log_points(
            f"{base}/wrist",
            [(visual_frame.wrist.x, visual_frame.wrist.y, visual_frame.wrist.z)],
            radius=self._config.wrist_radius,
            color=self._config.wrist_color,
        )
        points = visual_frame.landmarks.points
        if self._config.landmarks_are_wrist_relative:
            points = self._transform_landmarks_by_wrist(points=points, wrist=visual_frame.wrist)
        self._log_points(
            f"{base}/landmarks",
            points,
            radius=self._config.landmark_radius,
            color=self._landmark_color(visual_frame.side),
        )

    def log_event(self, event: ParsedPacket | HandFrame) -> None:
        """Log either packet or frame event.

        :param event:
            Stream event from :class:`hand_tracking_sdk.HTSClient`.
        """
        if isinstance(event, HandFrame):
            self.log_frame(event)
            return
        self.log_packet(event)

    def _log_points(
        self,
        path: str,
        points: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
        *,
        radius: float,
        color: tuple[int, int, int],
    ) -> None:
        mapped_points = [self._map_point_frame(x=x, y=y, z=z) for x, y, z in points]
        points_as_lists = [[x, y, z] for x, y, z in mapped_points]
        self._rr.log(
            path,
            self._rr.Points3D(
                points_as_lists,
                radii=[radius] * len(points_as_lists),
                colors=[list(color)] * len(points_as_lists),
            ),
        )

    def _import_rerun(self) -> ModuleType:
        try:
            module = importlib.import_module("rerun")
        except ModuleNotFoundError as exc:
            raise VisualizationDependencyError(
                "rerun is not installed. Install with: pip install hand-tracking-sdk[visualization]"
            ) from exc

        return module

    def _apply_view_background(self) -> None:
        """Apply optional background color to the default 3D view."""
        if self._config.background_color is None:
            return

        if not hasattr(self._rr, "send_blueprint"):
            return

        try:
            blueprint_module = importlib.import_module("rerun.blueprint")
        except ModuleNotFoundError:
            return

        blueprint = blueprint_module.Blueprint(
            blueprint_module.Spatial3DView(
                origin="/",
                name="3D Scene",
                background=list(self._config.background_color),
            )
        )
        self._rr.send_blueprint(blueprint)

    def _landmark_color(self, side: HandSide) -> tuple[int, int, int]:
        if side == HandSide.LEFT:
            return self._config.left_landmark_color
        return self._config.right_landmark_color

    def _map_point_frame(self, *, x: float, y: float, z: float) -> tuple[float, float, float]:
        if self._config.visualization_frame == VisualizationFrame.SDK:
            return (x, y, z)

        # Map SDK point basis to FLU: x=forward, y=left, z=up.
        return (x, z, -y)

    def _transform_landmarks_by_wrist(
        self,
        *,
        points: tuple[tuple[float, float, float], ...],
        wrist: WristPose,
    ) -> tuple[tuple[float, float, float], ...]:
        transformed: list[tuple[float, float, float]] = []
        for px, py, pz in points:
            rx, ry, rz = _rotate_vector_by_quaternion(
                x=px,
                y=py,
                z=pz,
                qx=wrist.qx,
                qy=wrist.qy,
                qz=wrist.qz,
                qw=wrist.qw,
            )
            transformed.append((rx + wrist.x, ry + wrist.y, rz + wrist.z))
        return tuple(transformed)


def _rotate_vector_by_quaternion(
    *,
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[float, float, float]:
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm == 0.0:
        return x, y, z
    qx_n = qx / norm
    qy_n = qy / norm
    qz_n = qz / norm
    qw_n = qw / norm

    # v' = q * v * q^-1
    ix = qw_n * x + qy_n * z - qz_n * y
    iy = qw_n * y + qz_n * x - qx_n * z
    iz = qw_n * z + qx_n * y - qy_n * x
    iw = -qx_n * x - qy_n * y - qz_n * z

    out_x = ix * qw_n + iw * -qx_n + iy * -qz_n - iz * -qy_n
    out_y = iy * qw_n + iw * -qy_n + iz * -qx_n - ix * -qz_n
    out_z = iz * qw_n + iw * -qz_n + ix * -qy_n - iy * -qx_n
    return cast(tuple[float, float, float], (out_x, out_y, out_z))
