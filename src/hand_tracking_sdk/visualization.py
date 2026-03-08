"""Optional real-time visualization helpers."""

from __future__ import annotations

import importlib
from collections import deque
from dataclasses import dataclass, field
from types import ModuleType
from typing import cast

from hand_tracking_sdk._compat import StrEnum
from hand_tracking_sdk.convert import (
    convert_hand_frame_unity_left_to_right,
    convert_landmarks_unity_left_to_right,
    convert_wrist_pose_unity_left_to_right,
    unity_left_to_right_position,
    unity_right_to_flu_position,
)
from hand_tracking_sdk.frame import HandFrame, HeadFrame
from hand_tracking_sdk.models import (
    HandSide,
    HeadPosePacket,
    LandmarksPacket,
    ParsedPacket,
    WristPacket,
    WristPose,
)

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
    :param show_jitter_panel:
        If ``True``, log jitter/drop scalar metrics under ``metrics/jitter/...``.
    :param jitter_window_size:
        Rolling window size for jitter percentile metrics.
    :param show_coordinate_frames:
        If ``True``, render local XYZ axes for wrist/head poses.
    :param coordinate_frame_axis_length:
        Axis length in meters for rendered coordinate frames.
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
    show_jitter_panel: bool = False
    jitter_window_size: int = 200
    show_coordinate_frames: bool = False
    coordinate_frame_axis_length: float = 0.08


@dataclass(slots=True)
class _JitterSideState:
    prev_recv_ts_ns: int | None = None
    prev_source_ts_ns: int | None = None
    prev_source_frame_seq: int | None = None
    jitter_ns_window: deque[int] = field(default_factory=deque)


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
        self._jitter_state_by_side: dict[HandSide, _JitterSideState] = {
            HandSide.LEFT: _JitterSideState(),
            HandSide.RIGHT: _JitterSideState(),
        }
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
            if self._config.show_coordinate_frames:
                self._log_pose_axes(
                    f"{side_path}/frame_axes",
                    position=(pose.x, pose.y, pose.z),
                    quaternion=(pose.qx, pose.qy, pose.qz, pose.qw),
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
            return

        if isinstance(packet, HeadPosePacket):
            hx, hy, hz = packet.data.x, packet.data.y, packet.data.z
            hqx, hqy, hqz, hqw = packet.data.qx, packet.data.qy, packet.data.qz, packet.data.qw
            if self._config.convert_to_right_handed:
                hx, hy, hz = unity_left_to_right_position(x=hx, y=hy, z=hz)
                as_wrist = WristPose(x=packet.data.x, y=packet.data.y, z=packet.data.z,
                                     qx=packet.data.qx, qy=packet.data.qy,
                                     qz=packet.data.qz, qw=packet.data.qw)
                converted = convert_wrist_pose_unity_left_to_right(as_wrist)
                hqx, hqy, hqz, hqw = converted.qx, converted.qy, converted.qz, converted.qw
            self._log_points(
                "head/pose",
                [(hx, hy, hz)],
                radius=self._config.wrist_radius,
                color=self._config.wrist_color,
            )
            if self._config.show_coordinate_frames:
                self._log_pose_axes(
                    "head/frame_axes",
                    position=(hx, hy, hz),
                    quaternion=(hqx, hqy, hqz, hqw),
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
        if self._config.show_coordinate_frames:
            self._log_pose_axes(
                f"{base}/frame_axes",
                position=(visual_frame.wrist.x, visual_frame.wrist.y, visual_frame.wrist.z),
                quaternion=(
                    visual_frame.wrist.qx,
                    visual_frame.wrist.qy,
                    visual_frame.wrist.qz,
                    visual_frame.wrist.qw,
                ),
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
        if self._config.show_jitter_panel:
            self._log_jitter_metrics(visual_frame)

    def log_event(self, event: ParsedPacket | HandFrame | HeadFrame) -> None:
        """Log either packet or frame event.

        :param event:
            Stream event from :class:`hand_tracking_sdk.HTSClient`.
        """
        if isinstance(event, HandFrame):
            self.log_frame(event)
            return
        if isinstance(event, HeadFrame):
            hx, hy, hz = event.head.x, event.head.y, event.head.z
            hqx, hqy, hqz, hqw = event.head.qx, event.head.qy, event.head.qz, event.head.qw
            if self._config.convert_to_right_handed:
                hx, hy, hz = unity_left_to_right_position(x=hx, y=hy, z=hz)
                as_wrist = WristPose(x=event.head.x, y=event.head.y, z=event.head.z,
                                     qx=event.head.qx, qy=event.head.qy,
                                     qz=event.head.qz, qw=event.head.qw)
                converted = convert_wrist_pose_unity_left_to_right(as_wrist)
                hqx, hqy, hqz, hqw = converted.qx, converted.qy, converted.qz, converted.qw
            self._log_points(
                f"frames/{event.frame_id}/head",
                [(hx, hy, hz)],
                radius=self._config.wrist_radius,
                color=self._config.wrist_color,
            )
            if self._config.show_coordinate_frames:
                self._log_pose_axes(
                    f"frames/{event.frame_id}/frame_axes",
                    position=(hx, hy, hz),
                    quaternion=(hqx, hqy, hqz, hqw),
                )
            return
        self.log_packet(event)

    def _log_pose_axes(
        self,
        path: str,
        *,
        position: tuple[float, float, float],
        quaternion: tuple[float, float, float, float],
    ) -> None:
        """Log XYZ axes for one pose with SDK-version compatibility.

        :param path:
            Entity path for axis geometry.
        :param position:
            Pose origin in SDK frame convention.
        :param quaternion:
            Pose orientation as ``(qx, qy, qz, qw)``.
        """
        px, py, pz = position
        qx, qy, qz, qw = quaternion
        axis_len = self._config.coordinate_frame_axis_length
        local_axes = self._coordinate_axes_for_visualization(axis_len=axis_len)
        axis_colors = ([255, 0, 0], [0, 255, 0], [0, 128, 255])
        origin = self._map_point_frame(x=px, y=py, z=pz)

        segments: list[list[list[float]]] = []
        for ax, ay, az in local_axes:
            rx, ry, rz = _rotate_vector_by_quaternion(
                x=ax,
                y=ay,
                z=az,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
            )
            endpoint = self._map_point_frame(x=px + rx, y=py + ry, z=pz + rz)
            segments.append(
                [
                    [origin[0], origin[1], origin[2]],
                    [endpoint[0], endpoint[1], endpoint[2]],
                ]
            )

        if hasattr(self._rr, "LineStrips3D"):
            try:
                self._rr.log(
                    path,
                    self._rr.LineStrips3D(
                        segments,
                        colors=axis_colors,
                    ),
                )
                return
            except Exception:
                pass

        fallback_points = [point for segment in segments for point in segment]
        fallback_colors = [
            axis_colors[0],
            axis_colors[0],
            axis_colors[1],
            axis_colors[1],
            axis_colors[2],
            axis_colors[2],
        ]
        self._rr.log(
            path,
            self._rr.Points3D(
                fallback_points,
                radii=[self._config.landmark_radius] * len(fallback_points),
                colors=fallback_colors,
            ),
        )

    def _coordinate_axes_for_visualization(
        self, *, axis_len: float
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """Return local axis vectors so displayed axes match configured view basis.

        In FLU mode, colors remain RGB while semantic directions become:
        red=+X(forward), green=+Y(left), blue=+Z(up).
        """
        if self._config.visualization_frame == VisualizationFrame.FLU:
            # Source (Unity-right) vectors that map to FLU +X/+Y/+Z after _map_point_frame.
            return (
                (0.0, 0.0, axis_len),   # FLU +X (forward)
                (-axis_len, 0.0, 0.0),  # FLU +Y (left)
                (0.0, -axis_len, 0.0),  # FLU +Z (up)
            )
        return (
            (axis_len, 0.0, 0.0),
            (0.0, axis_len, 0.0),
            (0.0, 0.0, axis_len),
        )

    def _log_points(
        self,
        path: str,
        points: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
        *,
        radius: float,
        color: tuple[int, int, int],
    ) -> None:
        """Log a homogeneous point set to Rerun.

        :param path:
            Entity path for the emitted points.
        :param points:
            Sequence of points in SDK frame convention prior to frame mapping.
        :param radius:
            Radius in meters for each point marker.
        :param color:
            RGB tuple used for all emitted points.
        """
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
        """Import the optional ``rerun`` dependency.

        :raises VisualizationDependencyError:
            If ``rerun-sdk`` is not installed.
        :returns:
            Imported ``rerun`` module.
        """
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

        eye_controls = None
        if hasattr(blueprint_module, "EyeControls3D"):
            eye_controls = blueprint_module.EyeControls3D(
                kind=blueprint_module.Eye3DKind.Orbital,
                position=[-0.06, 0.02, 1.5],
                look_target=[0.2, 0.02, 1.5],
                eye_up=[0.0, 0.0, 1.0],
            )

        spatial_view = blueprint_module.Spatial3DView(
            origin="/",
            name="3D Scene",
            background=list(self._config.background_color),
            eye_controls=eye_controls,
        )

        if not self._config.show_jitter_panel or not hasattr(blueprint_module, "TimeSeriesView"):
            self._rr.send_blueprint(blueprint_module.Blueprint(spatial_view))
            return

        try:
            jitter_view = blueprint_module.TimeSeriesView(
                origin="/metrics/jitter",
                name="Jitter",
            )
            if hasattr(blueprint_module, "Horizontal"):
                layout = blueprint_module.Horizontal(spatial_view, jitter_view)
            else:
                layout = [spatial_view, jitter_view]
            self._rr.send_blueprint(blueprint_module.Blueprint(layout))
        except Exception:
            # Fall back to plain 3D blueprint if viewer API shape differs by rerun version.
            self._rr.send_blueprint(blueprint_module.Blueprint(spatial_view))

    def _landmark_color(self, side: HandSide) -> tuple[int, int, int]:
        """Return the configured landmark color for a hand side.

        :param side:
            Hand side associated with the packet or frame.
        :returns:
            RGB color tuple for landmark points.
        """
        if side == HandSide.LEFT:
            return self._config.left_landmark_color
        return self._config.right_landmark_color

    def _map_point_frame(self, *, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Map a point from SDK coordinates to the visualization frame.

        :param x:
            Point X value in SDK frame.
        :param y:
            Point Y value in SDK frame.
        :param z:
            Point Z value in SDK frame.
        :returns:
            Point in the configured visualization frame.
        """
        if self._config.visualization_frame == VisualizationFrame.SDK:
            return (x, y, z)

        return unity_right_to_flu_position(x=x, y=y, z=z)

    def _transform_landmarks_by_wrist(
        self,
        *,
        points: tuple[tuple[float, float, float], ...],
        wrist: WristPose,
    ) -> tuple[tuple[float, float, float], ...]:
        """Transform wrist-relative landmarks into world coordinates.

        :param points:
            Landmark points interpreted as coordinates in wrist-local space.
        :param wrist:
            Wrist world pose used to rotate and translate each point.
        :returns:
            Landmark points in world coordinates.
        """
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

    def _log_jitter_metrics(self, frame: HandFrame) -> None:
        """Log per-side jitter and drop metrics as scalar timeseries."""
        if frame.source_ts_ns is None:
            return

        side = frame.side
        state = self._jitter_state_by_side[side]
        base = f"metrics/jitter/{side.value.lower()}"

        if state.prev_recv_ts_ns is not None and state.prev_source_ts_ns is not None:
            source_dt_ns = frame.source_ts_ns - state.prev_source_ts_ns
            recv_dt_ns = frame.recv_ts_ns - state.prev_recv_ts_ns
            jitter_ns = recv_dt_ns - source_dt_ns

            self._append_jitter_window(state.jitter_ns_window, jitter_ns)
            self._log_scalar(f"{base}/source_dt_ms", source_dt_ns / 1e6)
            self._log_scalar(f"{base}/recv_dt_ms", recv_dt_ns / 1e6)
            self._log_scalar(f"{base}/jitter_ms", jitter_ns / 1e6)
            self._log_scalar(
                f"{base}/jitter_p95_ms",
                self._percentile_ms(state.jitter_ns_window, 0.95),
            )

        if frame.source_frame_seq is not None and state.prev_source_frame_seq is not None:
            dropped = max(0, frame.source_frame_seq - state.prev_source_frame_seq - 1)
            self._log_scalar(f"{base}/drop_gap_frames", float(dropped))

        state.prev_recv_ts_ns = frame.recv_ts_ns
        state.prev_source_ts_ns = frame.source_ts_ns
        state.prev_source_frame_seq = frame.source_frame_seq

    def _append_jitter_window(self, window: deque[int], value: int) -> None:
        """Append to a bounded jitter window."""
        window.append(value)
        while len(window) > self._config.jitter_window_size:
            window.popleft()

    def _percentile_ms(self, values_ns: deque[int], fraction: float) -> float:
        """Return percentile value in milliseconds for a small rolling window."""
        if not values_ns:
            return 0.0
        sorted_values = sorted(values_ns)
        index = int(round((len(sorted_values) - 1) * fraction))
        return sorted_values[index] / 1e6

    def _log_scalar(self, path: str, value: float) -> None:
        """Log one scalar point, with compatibility across rerun SDK versions."""
        if hasattr(self._rr, "Scalar"):
            self._rr.log(path, self._rr.Scalar(value))
            return
        if hasattr(self._rr, "Scalars"):
            self._rr.log(path, self._rr.Scalars([value]))


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
    """Rotate a 3D vector by a quaternion.

    :param x:
        Input vector X component.
    :param y:
        Input vector Y component.
    :param z:
        Input vector Z component.
    :param qx:
        Quaternion X component.
    :param qy:
        Quaternion Y component.
    :param qz:
        Quaternion Z component.
    :param qw:
        Quaternion W component.
    :returns:
        Rotated ``(x, y, z)`` tuple.
    """
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
