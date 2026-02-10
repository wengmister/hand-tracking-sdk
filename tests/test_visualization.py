from __future__ import annotations

from types import ModuleType

import pytest

from hand_tracking_sdk import (
    HandFrame,
    HandLandmarks,
    HandSide,
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


class _FakeBlueprint(ModuleType):
    class Spatial3DView:
        def __init__(
            self,
            *,
            origin: str,
            name: str,
            background: list[int],
        ) -> None:
            self.origin = origin
            self.name = name
            self.background = background

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

    assert fake.inits == [("hts-test", False)]
    assert any(path == "hands/left/wrist" for path, _ in fake.logs)
    assert any(path == "hands/left/landmarks" for path, _ in fake.logs)
    assert any(path == "frames/left_hand_link/wrist" for path, _ in fake.logs)
    assert any(path == "frames/left_hand_link/landmarks" for path, _ in fake.logs)


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
            landmarks_are_wrist_relative=True,
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
