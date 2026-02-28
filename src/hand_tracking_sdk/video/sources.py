"""Video source adapters for host-side WebRTC transmission."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, monotonic_ns
from typing import Any


@dataclass(frozen=True, slots=True)
class VideoFormat:
    """Video format contract produced by a source adapter."""

    width: int
    height: int
    fps: int


class VideoSourceAdapter(ABC):
    """Abstract source adapter used by the WebRTC sender."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize source resources."""

    @abstractmethod
    async def stop(self) -> None:
        """Release source resources."""

    @abstractmethod
    async def next_frame(self) -> Any:
        """Return next video frame object compatible with `av.VideoFrame`."""

    @abstractmethod
    def get_format(self) -> VideoFormat:
        """Return source format."""


class TestPatternSourceAdapter(VideoSourceAdapter):
    """Synthetic color-bar test source."""

    def __init__(self, *, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self._format = VideoFormat(width=width, height=height, fps=fps)
        self._frame_index = 0

    async def start(self) -> None:
        self._frame_index = 0

    async def stop(self) -> None:
        return None

    def get_format(self) -> VideoFormat:
        return self._format

    async def next_frame(self) -> Any:
        # Lazy imports keep video dependencies optional for non-video SDK usage.
        import av
        import numpy as np

        w, h = self._format.width, self._format.height
        image = np.zeros((h, w, 3), dtype=np.uint8)
        t = self._frame_index

        # Horizontal RGB bars + moving luminance stripe.
        band_w = max(1, w // 6)
        image[:, 0:band_w, :] = (255, 0, 0)
        image[:, band_w : 2 * band_w, :] = (0, 255, 0)
        image[:, 2 * band_w : 3 * band_w, :] = (0, 0, 255)
        image[:, 3 * band_w : 4 * band_w, :] = (255, 255, 0)
        image[:, 4 * band_w : 5 * band_w, :] = (255, 0, 255)
        image[:, 5 * band_w :, :] = (0, 255, 255)
        stripe_x = (t * 7) % w
        image[:, max(0, stripe_x - 8) : min(w, stripe_x + 8), :] = (255, 255, 255)

        # Encode frame index timestamp pattern as low-cost modulation.
        pulse = int((monotonic_ns() // 100_000_000) % 2) * 30
        image[0:20, 0:200, :] = (pulse, pulse, pulse)

        self._frame_index += 1
        return av.VideoFrame.from_ndarray(image, format="rgb24")


class WebcamSourceAdapter(VideoSourceAdapter):
    """Webcam-backed source adapter."""

    def __init__(
        self,
        *,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self._format = VideoFormat(width=width, height=height, fps=fps)
        self._device_index = device_index
        self._capture: Any = None

    async def start(self) -> None:
        import cv2

        capture = cv2.VideoCapture(self._device_index)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open webcam index {self._device_index}.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._format.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._format.height))
        capture.set(cv2.CAP_PROP_FPS, float(self._format.fps))
        self._capture = capture

    async def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def get_format(self) -> VideoFormat:
        return self._format

    async def next_frame(self) -> Any:
        import av
        import cv2

        if self._capture is None:
            raise RuntimeError("Webcam source not started.")

        ok, frame_bgr = self._capture.read()
        if not ok:
            raise RuntimeError("Failed to read webcam frame.")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")


class MujocoSourceAdapter(VideoSourceAdapter):
    """MuJoCo offscreen renderer source adapter."""

    def __init__(
        self,
        *,
        model_path: str,
        camera: str | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        pre_step: Callable[[Any, Any], None] | None = None,
    ) -> None:
        self._format = VideoFormat(width=width, height=height, fps=fps)
        self._model_path = model_path
        self._camera = camera
        self._camera_arg: Any = camera
        self._pre_step = pre_step
        self._mujoco: Any = None
        self._model: Any = None
        self._data: Any = None
        self._renderer: Any = None
        self._last_frame_ts = 0.0

    async def start(self) -> None:
        try:
            import mujoco
        except Exception as exc:
            raise RuntimeError(
                "mujoco is required for sim source. "
                "Install with: pip install hand-tracking-sdk[sim]"
            ) from exc

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(self._model_path)
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(
            self._model,
            width=self._format.width,
            height=self._format.height,
        )
        if self._camera is not None and self._camera.isdigit():
            self._camera_arg = int(self._camera)
        mujoco.mj_forward(self._model, self._data)
        self._last_frame_ts = monotonic()

    async def stop(self) -> None:
        if self._renderer is not None:
            close_fn = getattr(self._renderer, "close", None)
            if callable(close_fn):
                close_fn()
        self._renderer = None
        self._data = None
        self._model = None
        self._mujoco = None

    def get_format(self) -> VideoFormat:
        return self._format

    async def next_frame(self) -> Any:
        import av

        if (
            self._mujoco is None
            or self._model is None
            or self._data is None
            or self._renderer is None
        ):
            raise RuntimeError("MuJoCo source not started.")

        # Basic frame pacing to target FPS.
        frame_interval_s = 1.0 / max(1, self._format.fps)
        now = monotonic()
        sleep_s = frame_interval_s - (now - self._last_frame_ts)
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
        self._last_frame_ts = monotonic()

        # Apply external inputs (e.g. mocap poses) then step physics.
        if self._pre_step is not None:
            self._pre_step(self._model, self._data)
        self._mujoco.mj_step(self._model, self._data)
        self._renderer.update_scene(self._data, camera=self._camera_arg)
        frame_rgb = self._renderer.render()
        return av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
