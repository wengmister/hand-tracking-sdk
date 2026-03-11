from __future__ import annotations

from types import ModuleType

import pytest

from hand_tracking_sdk import (
    HandFrame,
    HandLandmarks,
    HandSide,
    HeadFrame,
    HeadPose,
    HeadPosePacket,
    LandmarksPacket,
    PacketType,
    RerunVisualizer,
    RerunVisualizerConfig,
    VisualizationDependencyError,
    WristPacket,
    WristPose,
)


class _FakeRerun(ModuleType):
    def __init__(self) -> None:
        super().__init__("rerun")
        self.inits: list[tuple[str, bool]] = []
        self.logs: list[tuple[str, object]] = []
        self.blueprints: list[object] = []

    def init(self, application_id: str, *, spawn: bool) -> None:
        self.inits.append((application_id, spawn))

    def log(self, path: str, payload: object) -> None:
        self.logs.append((path, payload))

    def send_blueprint(self, blueprint: object) -> None:
        self.blueprints.append(blueprint)

    class Points3D:
        def __init__(
            self,
            points: list[list[float]],
            *,
            radii: list[float] | None = None,
            colors: list[list[int]] | None = None,
        ) -> None:
            self.points = points
            self.radii = radii
            self.colors = colors

    class Scalar:
        def __init__(self, value: float) -> None:
            self.value = value

    class LineStrips3D:
        def __init__(
            self,
            strips: list[list[list[float]]],
            *,
            colors: list[list[int]] | None = None,
        ) -> None:
            self.strips = strips
            self.colors = colors


class _FakeBlueprint(ModuleType):
    class Spatial3DView:
        def __init__(
            self,
            *,
            origin: str,
            name: str,
            background: list[int],
            eye_controls: object | None = None,
        ) -> None:
            self.origin = origin
            self.name = name
            self.background = background
            self.eye_controls = eye_controls

    class Blueprint:
        def __init__(self, view: object) -> None:
            self.view = view


def test_rerun_visualizer_requires_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_module_not_found(_: str) -> ModuleType:
        raise ModuleNotFoundError("rerun")

    monkeypatch.setattr("importlib.import_module", _raise_module_not_found)

    with pytest.raises(VisualizationDependencyError):
        RerunVisualizer()


def test_rerun_visualizer_logs_packet_and_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRerun()
    fake_blueprint = _FakeBlueprint("rerun.blueprint")

    def _import(module_name: str) -> ModuleType:
        if module_name == "rerun":
            return fake
        if module_name == "rerun.blueprint":
            return fake_blueprint
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr("importlib.import_module", _import)

    visualizer = RerunVisualizer(RerunVisualizerConfig(application_id="hts-test", spawn=False))

    packet = WristPacket(
        side=HandSide.LEFT,
        kind=PacketType.WRIST,
        data=WristPose(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
    )
    landmarks_packet = LandmarksPacket(
        side=HandSide.LEFT,
        kind=PacketType.LANDMARKS,
        data=HandLandmarks(points=((1.0, 2.0, 3.0),)),
    )
    frame = HandFrame(
        side=HandSide.LEFT,
        frame_id="left_hand_link",
        wrist=WristPose(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        landmarks=HandLandmarks(points=((1.0, 2.0, 3.0),)),
        sequence_id=1,
        recv_ts_ns=100,
        recv_time_unix_ns=200,
        source_ts_ns=None,
        wrist_recv_ts_ns=90,
        landmarks_recv_ts_ns=95,
    )

    visualizer.log_packet(packet)
    visualizer.log_packet(landmarks_packet)
    visualizer.log_frame(frame)
    visualizer.log_packet(
        HeadPosePacket(
            side=HandSide.HEAD,
            kind=PacketType.POSE,
            data=HeadPose(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        )
    )
    visualizer.log_event(
        HeadFrame(
            side=HandSide.HEAD,
            frame_id="head_link",
            head=HeadPose(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
            sequence_id=0,
            recv_ts_ns=100,
            recv_time_unix_ns=200,
            source_ts_ns=None,
        )
    )

    assert fake.inits == [("hts-test", False)]
    assert any(path == "hands/left/wrist" for path, _ in fake.logs)
    assert any(path == "hands/left/landmarks" for path, _ in fake.logs)
    assert any(path == "frames/left_hand_link/wrist" for path, _ in fake.logs)
    assert any(path == "frames/left_hand_link/landmarks" for path, _ in fake.logs)
    assert any(path == "head/pose" for path, _ in fake.logs)
    assert any(path == "frames/head_link/head" for path, _ in fake.logs)


def test_coordinate_frames_are_logged_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRerun()
    fake_blueprint = _FakeBlueprint("rerun.blueprint")

    def _import(module_name: str) -> ModuleType:
        if module_name == "rerun":
            return fake
        if module_name == "rerun.blueprint":
            return fake_blueprint
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr("importlib.import_module", _import)

    visualizer = RerunVisualizer(
        RerunVisualizerConfig(
            application_id="hts-test",
            spawn=False,
            show_coordinate_frames=True,
        )
    )

    visualizer.log_packet(
        WristPacket(
            side=HandSide.LEFT,
            kind=PacketType.WRIST,
            data=WristPose(x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        )
    )
    visualizer.log_packet(
        HeadPosePacket(
            side=HandSide.HEAD,
            kind=PacketType.POSE,
            data=HeadPose(x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        )
    )

    logged_paths = [path for path, _ in fake.logs]
    assert "hands/left/frame_axes" in logged_paths
    assert "head/frame_axes" in logged_paths
    frame_payload = next(payload for path, payload in fake.logs if path == "hands/left/frame_axes")
    assert isinstance(frame_payload, _FakeRerun.LineStrips3D)
    # In FLU view, RGB axes should map to +X/+Y/+Z respectively.
    assert frame_payload.strips[0][1] == [0.08, -0.0, -0.0]  # red -> +X (forward)
    assert frame_payload.strips[1][1] == [0.0, 0.08, -0.0]  # green -> +Y (left)
    assert frame_payload.strips[2][1] == [0.0, -0.0, 0.08]  # blue -> +Z (up)


def test_landmarks_are_transformed_by_wrist_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRerun()
    fake_blueprint = _FakeBlueprint("rerun.blueprint")

    def _import(module_name: str) -> ModuleType:
        if module_name == "rerun":
            return fake
        if module_name == "rerun.blueprint":
            return fake_blueprint
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr("importlib.import_module", _import)

    visualizer = RerunVisualizer(
        RerunVisualizerConfig(
            application_id="hts-test",
            spawn=False,
        )
    )

    wrist_packet = WristPacket(
        side=HandSide.LEFT,
        kind=PacketType.WRIST,
        data=WristPose(x=10.0, y=20.0, z=30.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
    )
    landmarks_packet = LandmarksPacket(
        side=HandSide.LEFT,
        kind=PacketType.LANDMARKS,
        data=HandLandmarks(points=((1.0, 2.0, 3.0),)),
    )

    visualizer.log_packet(wrist_packet)
    visualizer.log_packet(landmarks_packet)

    landmarks_payload = next(
        payload for path, payload in fake.logs if path == "hands/left/landmarks"
    )
    assert isinstance(landmarks_payload, _FakeRerun.Points3D)
    assert landmarks_payload.points == [[33.0, -11.0, 22.0]]
    assert landmarks_payload.radii == [0.015]
    assert landmarks_payload.colors == [[64, 128, 255]]


def test_right_landmarks_use_red_color(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRerun()
    fake_blueprint = _FakeBlueprint("rerun.blueprint")

    def _import(module_name: str) -> ModuleType:
        if module_name == "rerun":
            return fake
        if module_name == "rerun.blueprint":
            return fake_blueprint
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr("importlib.import_module", _import)

    visualizer = RerunVisualizer(RerunVisualizerConfig(application_id="hts-test", spawn=False))
    visualizer.log_packet(
        WristPacket(
            side=HandSide.RIGHT,
            kind=PacketType.WRIST,
            data=WristPose(x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        )
    )
    visualizer.log_packet(
        LandmarksPacket(
            side=HandSide.RIGHT,
            kind=PacketType.LANDMARKS,
            data=HandLandmarks(points=((0.0, 0.0, 0.0),)),
        )
    )

    landmarks_payload = next(
        payload for path, payload in fake.logs if path == "hands/right/landmarks"
    )
    assert isinstance(landmarks_payload, _FakeRerun.Points3D)
    assert landmarks_payload.colors == [[255, 64, 64]]


def test_default_visualizer_config_values() -> None:
    config = RerunVisualizerConfig()

    assert config.application_id == "hand-tracking-sdk"
    assert config.spawn is True
    assert config.landmarks_are_wrist_relative is True
    assert config.show_jitter_panel is False
    assert config.show_coordinate_frames is False


def test_jitter_metrics_are_logged_from_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRerun()
    fake_blueprint = _FakeBlueprint("rerun.blueprint")

    def _import(module_name: str) -> ModuleType:
        if module_name == "rerun":
            return fake
        if module_name == "rerun.blueprint":
            return fake_blueprint
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr("importlib.import_module", _import)
    visualizer = RerunVisualizer(
        RerunVisualizerConfig(
            application_id="hts-test",
            spawn=False,
            show_jitter_panel=True,
        )
    )

    frame_1 = HandFrame(
        side=HandSide.RIGHT,
        frame_id="right_hand_link",
        wrist=WristPose(x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        landmarks=HandLandmarks(points=((0.0, 0.0, 0.0),)),
        sequence_id=0,
        recv_ts_ns=1_000_000_000,
        recv_time_unix_ns=2_000_000_000,
        source_ts_ns=500_000_000,
        wrist_recv_ts_ns=900_000_000,
        landmarks_recv_ts_ns=950_000_000,
        source_frame_seq=10,
    )
    frame_2 = HandFrame(
        side=HandSide.RIGHT,
        frame_id="right_hand_link",
        wrist=WristPose(x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        landmarks=HandLandmarks(points=((0.0, 0.0, 0.0),)),
        sequence_id=1,
        recv_ts_ns=1_020_000_000,
        recv_time_unix_ns=2_020_000_000,
        source_ts_ns=510_000_000,
        wrist_recv_ts_ns=1_010_000_000,
        landmarks_recv_ts_ns=1_015_000_000,
        source_frame_seq=12,
    )

    visualizer.log_frame(frame_1)
    visualizer.log_frame(frame_2)

    logged_paths = [path for path, _ in fake.logs]
    assert "metrics/jitter/right/source_dt_ms" in logged_paths
    assert "metrics/jitter/right/recv_dt_ms" in logged_paths
    assert "metrics/jitter/right/jitter_ms" in logged_paths
    assert "metrics/jitter/right/jitter_p95_ms" in logged_paths
    assert "metrics/jitter/right/drop_gap_frames" in logged_paths
