"""High-level host video service orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hand_tracking_sdk.video.schemas import SignalingMessage, make_signaling_message
from hand_tracking_sdk.video.signaling import SignalingConnection, VideoSignalingServer
from hand_tracking_sdk.video.sources import (
    MujocoSourceAdapter,
    TestPatternSourceAdapter,
    VideoSourceAdapter,
    WebcamSourceAdapter,
)
from hand_tracking_sdk.video.webrtc_sender import VideoSenderStats, VideoWebRTCSender


@dataclass(frozen=True, slots=True)
class VideoServiceConfig:
    """Configuration for `VideoService` runtime."""

    signaling_host: str = "0.0.0.0"
    signaling_port: int = 8765
    source: str = "test"
    preset: str = "720p"
    webcam_index: int = 0
    mj_model_path: str | None = None
    mj_camera: str | None = None
    mj_pre_step: Callable[[Any, Any], None] | None = None
    mj_perf_hook: Callable[[dict[str, float]], None] | None = None
    stats_interval_s: float = 1.0
    server_version: str = "0.1.0"
    verbose: bool = False
    log_hook: Callable[[str], None] | None = None


class VideoService:
    """Owns signaling + sender + source lifecycle for one video session."""

    def __init__(
        self,
        config: VideoServiceConfig | None = None,
        *,
        sender_factory: Callable[[VideoSourceAdapter, str, int], VideoWebRTCSender] | None = None,
    ) -> None:
        """Initialize service dependencies and runtime state."""
        self._config = config or VideoServiceConfig()
        self._sender_factory = sender_factory or self._default_sender_factory
        self._signaling = VideoSignalingServer(
            host=self._config.signaling_host,
            port=self._config.signaling_port,
            on_message=self._on_message,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
        )
        self._sender: VideoWebRTCSender | None = None
        self._active_connection: SignalingConnection | None = None
        self._active_session_id: str | None = None
        self._stats_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start signaling server."""
        await self._signaling.start()
        self._log(
            "signaling server listening "
            f"host={self._config.signaling_host} "
            f"port={self._config.signaling_port}"
        )

    async def stop(self) -> None:
        """Stop full service and active session state."""
        await self._stop_sender()
        if self._stats_task is not None:
            self._stats_task.cancel()
            self._stats_task = None
        await self._signaling.stop()
        self._active_connection = None
        self._active_session_id = None
        self._log("video service stopped")

    async def _on_message(self, connection: SignalingConnection, message: SignalingMessage) -> None:
        session_id = message.session_id
        message_type = message.type
        self._log(f"recv type={message_type} session={session_id}")

        if message_type == "hello":
            await self._signaling.send(
                connection,
                make_signaling_message(
                    type="hello_ack",
                    session_id=session_id,
                    payload={"server_version": self._config.server_version},
                ),
            )
            self._log(f"sent hello_ack session={session_id}")
            return

        if message_type == "ping":
            await self._signaling.send(
                connection,
                make_signaling_message(type="pong", session_id=session_id, payload=message.payload),
            )
            self._log(f"sent pong session={session_id}")
            return

        if message_type == "start_video":
            self._active_connection = connection
            self._active_session_id = session_id
            await self._send_video_state(connection, session_id=session_id, state="connecting")
            self._log(f"video_state connecting session={session_id}")
            return

        if message_type == "stop_video":
            await self._stop_sender()
            await self._send_video_state(connection, session_id=session_id, state="stopped")
            self._log(f"video_state stopped session={session_id}")
            return

        if message_type == "offer":
            await self._handle_offer(connection, message)
            return

        if message_type == "ice_candidate":
            await self._handle_remote_ice_candidate(connection, message)
            return

        await self._signaling.send(
            connection,
            make_signaling_message(
                type="error",
                session_id=session_id,
                payload={
                    "code": "unsupported_message",
                    "message": f"Unsupported message: {message_type}",
                },
            ),
        )
        self._log(f"unsupported message type={message_type} session={session_id}")

    async def _handle_offer(
        self,
        connection: SignalingConnection,
        message: SignalingMessage,
    ) -> None:
        session_id = message.session_id
        sdp = str(message.payload.get("sdp", ""))
        if not sdp:
            await self._emit_error(
                connection,
                session_id,
                "missing_offer",
                "Offer payload missing 'sdp'.",
            )
            self._log(f"missing offer SDP session={session_id}")
            return
        if "m=video" not in sdp:
            await self._emit_error(
                connection,
                session_id,
                "invalid_offer",
                "Offer missing m=video; Quest must create recv-only "
                "video transceiver before CreateOffer.",
            )
            self._log(f"invalid offer (no m=video) session={session_id} sdp_len={len(sdp)}")
            return
        self._log(f"offer received session={session_id} sdp_len={len(sdp)}")

        if self._sender is None:
            try:
                source = self._build_source()
                self._sender = self._sender_factory(
                    source,
                    session_id,
                    self._parse_fps(self._config.preset),
                )
                await self._sender.start()
                self._log(
                    f"sender started source={self._config.source} "
                    f"preset={self._config.preset}"
                )
            except Exception as exc:
                await self._emit_error(connection, session_id, "sender_start_failed", str(exc))
                await self._send_video_state(
                    connection,
                    session_id=session_id,
                    state="error",
                    reason="sender_start_failed",
                )
                self._log(f"sender start failed session={session_id}: {exc}")
                return

        try:
            answer_sdp = await self._sender.apply_offer(sdp_offer=sdp)
        except Exception as exc:
            await self._emit_error(connection, session_id, "offer_failed", str(exc))
            await self._send_video_state(
                connection,
                session_id=session_id,
                state="error",
                reason="offer_failed",
            )
            await self._stop_sender()
            self._log(f"offer handling failed session={session_id}: {exc}")
            return

        await self._signaling.send(
            connection,
            make_signaling_message(
                type="answer",
                session_id=session_id,
                payload={"sdp": answer_sdp},
            ),
        )
        self._log(f"answer sent session={session_id} sdp_len={len(answer_sdp)}")
        await self._send_video_state(connection, session_id=session_id, state="playing")
        self._log(f"video_state playing session={session_id}")
        self._start_stats_loop_if_needed()

    async def _handle_remote_ice_candidate(
        self,
        connection: SignalingConnection,
        message: SignalingMessage,
    ) -> None:
        if self._sender is None:
            await self._emit_error(
                connection,
                message.session_id,
                "no_sender",
                "Received ICE candidate before sender initialization.",
            )
            return

        try:
            await self._sender.add_remote_ice_candidate(
                candidate=str(message.payload.get("candidate", "")),
                sdp_mid=self._to_optional_str(message.payload.get("sdpMid")),
                sdp_mline_index=self._to_optional_int(message.payload.get("sdpMLineIndex")),
            )
            self._log(f"remote ICE applied session={message.session_id}")
        except Exception as exc:
            await self._emit_error(connection, message.session_id, "ice_failed", str(exc))
            self._log(f"remote ICE failed session={message.session_id}: {exc}")

    async def _on_disconnect(self, _: SignalingConnection) -> None:
        await self._stop_sender()
        self._log("client disconnected; sender stopped")

    async def _on_connect(self, connection: SignalingConnection) -> None:
        remote = getattr(connection.websocket, "remote_address", None)
        self._log(f"client connected remote={remote}")

    def _start_stats_loop_if_needed(self) -> None:
        if self._stats_task is None or self._stats_task.done():
            self._stats_task = asyncio.create_task(self._stats_loop())

    async def _stats_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.stats_interval_s)
            if (
                self._sender is None
                or self._active_connection is None
                or self._active_session_id is None
            ):
                continue

            try:
                stats = await self._sender.get_stats()
                await self._emit_stats(self._active_connection, self._active_session_id, stats)
                self._log(
                    "stats "
                    f"session={self._active_session_id} "
                    f"fps={stats.fps:.1f} bitrate_kbps={stats.bitrate_kbps:.1f} "
                    f"drops={stats.frame_drops} rtt_ms={stats.rtt_ms}"
                )
            except Exception:
                continue

    async def _emit_stats(
        self,
        connection: SignalingConnection,
        session_id: str,
        stats: VideoSenderStats,
    ) -> None:
        await self._signaling.send(
            connection,
            make_signaling_message(
                type="stats",
                session_id=session_id,
                payload={
                    "fps": round(stats.fps, 2),
                    "bitrate_kbps": round(stats.bitrate_kbps, 2),
                    "frame_drops": stats.frame_drops,
                    "rtt_ms": None if stats.rtt_ms is None else round(stats.rtt_ms, 2),
                },
            ),
        )

    async def _emit_error(
        self,
        connection: SignalingConnection,
        session_id: str,
        code: str,
        message: str,
    ) -> None:
        await self._signaling.send(
            connection,
            make_signaling_message(
                type="error",
                session_id=session_id,
                payload={"code": code, "message": message},
            ),
        )
        self._log(f"error session={session_id} code={code} message={message}")

    async def _send_video_state(
        self,
        connection: SignalingConnection,
        *,
        session_id: str,
        state: str,
        reason: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"state": state}
        if reason is not None:
            payload["reason"] = reason
        await self._signaling.send(
            connection,
            make_signaling_message(type="video_state", session_id=session_id, payload=payload),
        )

    async def _stop_sender(self) -> None:
        if self._sender is not None:
            await self._sender.stop()
            self._sender = None
            self._log("sender stopped")

    _VALID_SOURCES = ("test", "webcam", "mujoco")

    def _build_source(self) -> VideoSourceAdapter:
        width, height, fps = self._parse_preset(self._config.preset)
        source = self._config.source
        if source not in self._VALID_SOURCES:
            raise ValueError(
                f"Unknown source {source!r}; "
                f"expected one of {self._VALID_SOURCES}"
            )
        if source == "mujoco":
            if not self._config.mj_model_path:
                raise ValueError("mujoco source requires mj_model_path (--mj-model).")
            return MujocoSourceAdapter(
                model_path=self._config.mj_model_path,
                camera=self._config.mj_camera,
                width=width,
                height=height,
                fps=fps,
                pre_step=self._config.mj_pre_step,
                perf_hook=self._config.mj_perf_hook,
            )
        if source == "webcam":
            return WebcamSourceAdapter(
                device_index=self._config.webcam_index,
                width=width,
                height=height,
                fps=fps,
            )
        return TestPatternSourceAdapter(width=width, height=height, fps=fps)

    def _default_sender_factory(
        self,
        source: VideoSourceAdapter,
        session_id: str,
        _: int,
    ) -> VideoWebRTCSender:
        async def _on_candidate(payload: dict[str, Any]) -> None:
            if self._active_connection is None:
                return
            await self._signaling.send(
                self._active_connection,
                make_signaling_message(
                    type="ice_candidate",
                    session_id=session_id,
                    payload=payload,
                ),
            )

        return VideoWebRTCSender(
            source=source,
            on_local_ice_candidate=_on_candidate,
            log_hook=lambda msg: self._log(f"[sender] {msg}"),
        )

    _VALID_PRESETS = ("480p", "720p", "1080p")

    # FPS is best-effort (software encoding is the bottleneck).  The value
    # here only sets the RTP time_base for timestamp calculation.
    _PRESET_MAP: dict[str, tuple[int, int, int]] = {
        "480p": (640, 480, 60),
        "720p": (1280, 720, 60),
        "1080p": (1920, 1080, 60),
    }

    def _parse_preset(self, preset: str) -> tuple[int, int, int]:
        normalized = preset.lower()
        result = self._PRESET_MAP.get(normalized)
        if result is None:
            raise ValueError(
                f"Unknown preset {preset!r}; "
                f"expected one of {self._VALID_PRESETS}"
            )
        return result

    def _parse_fps(self, preset: str) -> int:
        return self._parse_preset(preset)[2]

    def _to_optional_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _to_optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        return text if text else None

    def _log(self, message: str) -> None:
        if not self._config.verbose:
            return
        if self._config.log_hook is not None:
            self._config.log_hook(message)
            return
        print(f"[video-service] {message}")
