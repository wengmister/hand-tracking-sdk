"""Video source adapters for host-side WebRTC transmission."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import monotonic_ns
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
