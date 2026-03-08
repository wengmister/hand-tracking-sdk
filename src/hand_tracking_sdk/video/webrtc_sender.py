"""Host-side WebRTC sender for one outbound H.264 video track."""

from __future__ import annotations

import asyncio
import fractions
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any

from hand_tracking_sdk.video.sources import VideoSourceAdapter


@dataclass(frozen=True, slots=True)
class VideoSenderStats:
    """Observable sender-side stats snapshot."""

    fps: float
    bitrate_kbps: float
    frame_drops: int
    rtt_ms: float | None


class _AdapterVideoTrack:  # Runtime subclass after aiortc import.
    """Internal adapter used to bridge source frames into aiortc track API."""

    kind = "video"

    def __init__(self, source: VideoSourceAdapter, fps: int) -> None:
        self._source = source
        self._fps = max(1, fps)
        self._pts = 0
        self._time_base = fractions.Fraction(1, self._fps)

    async def recv(self) -> Any:
        frame = await self._source.next_frame()
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 1
        return frame


class VideoWebRTCSender:
    """One-to-one sender peer for host->Quest video."""

    def __init__(
        self,
        *,
        source: VideoSourceAdapter,
        on_local_ice_candidate: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        log_hook: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize sender with a frame source and optional ICE callback."""
        self._source = source
        self._on_local_ice_candidate = on_local_ice_candidate
        self._log_hook = log_hook
        self._pc: Any = None
        self._created_at = monotonic()
        self._frames_sent = 0
        self._bytes_sent = 0
        self._frame_drops = 0
        self._last_stats_lock = asyncio.Lock()
        self._h264_forced = False

    async def start(self) -> None:
        """Start the source and create the outbound peer/video track."""
        await self._source.start()
        self._pc = self._new_peer_connection()
        self._add_video_track()
        self._wire_ice_callbacks()

    async def stop(self) -> None:
        """Stop peer connection and release source resources."""
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
        await self._source.stop()

    async def apply_offer(self, *, sdp_offer: str) -> str:
        """Apply Quest SDP offer and return host SDP answer."""
        if self._pc is None:
            raise RuntimeError("Video sender not started.")

        rtc_session_description = self._import_aiortc_symbol("RTCSessionDescription")
        offer = rtc_session_description(sdp=sdp_offer, type="offer")
        await self._pc.setRemoteDescription(offer)
        self._force_h264_codec_if_possible()
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        for t in self._pc.getTransceivers():
            self._log(
                f"transceiver kind={t.kind} direction={t.direction} "
                f"currentDirection={t.currentDirection}"
            )
        return str(self._pc.localDescription.sdp)

    async def add_remote_ice_candidate(
        self,
        *,
        candidate: str,
        sdp_mid: str | None,
        sdp_mline_index: int | None,
    ) -> None:
        """Apply one remote ICE candidate from Quest."""
        if self._pc is None:
            return
        if not candidate:
            return

        candidate_from_sdp = self._import_aiortc_sdp_symbol("candidate_from_sdp")
        parsed_candidate = candidate_from_sdp(candidate)
        parsed_candidate.sdpMid = sdp_mid
        parsed_candidate.sdpMLineIndex = sdp_mline_index
        await self._pc.addIceCandidate(parsed_candidate)

    async def get_stats(self) -> VideoSenderStats:
        """Return sender stats snapshot."""
        if self._pc is None:
            return VideoSenderStats(fps=0.0, bitrate_kbps=0.0, frame_drops=0, rtt_ms=None)

        async with self._last_stats_lock:
            try:
                report = await self._pc.getStats()
            except Exception:
                report = {}

            bytes_sent = self._bytes_sent
            frames_sent = self._frames_sent
            rtt_ms: float | None = None
            for stat in report.values() if hasattr(report, "values") else []:
                stat_type = getattr(stat, "type", "")
                if stat_type == "outbound-rtp":
                    bytes_sent = int(getattr(stat, "bytesSent", bytes_sent))
                    frames_sent = int(getattr(stat, "framesSent", frames_sent))
                if stat_type == "candidate-pair":
                    current_rtt = getattr(stat, "currentRoundTripTime", None)
                    if current_rtt is not None:
                        rtt_ms = float(current_rtt) * 1000.0

            elapsed_s = max(0.001, monotonic() - self._created_at)
            bitrate_kbps = (max(0, bytes_sent) * 8.0 / elapsed_s) / 1000.0
            fps = max(0, frames_sent) / elapsed_s
            self._bytes_sent = bytes_sent
            self._frames_sent = frames_sent
            return VideoSenderStats(
                fps=fps,
                bitrate_kbps=bitrate_kbps,
                frame_drops=self._frame_drops,
                rtt_ms=rtt_ms,
            )

    def _new_peer_connection(self) -> Any:
        rtc_peer_connection = self._import_aiortc_symbol("RTCPeerConnection")
        rtc_configuration = self._import_aiortc_symbol("RTCConfiguration")
        rtc_ice_server = self._import_aiortc_symbol("RTCIceServer")
        config = rtc_configuration(
            iceServers=[
                rtc_ice_server(urls=["stun:stun.l.google.com:19302"]),
            ]
        )
        pc = rtc_peer_connection(config)
        self._wire_connection_state(pc)
        return pc

    def _wire_connection_state(self, pc: Any) -> None:
        @pc.on("connectionstatechange")  # type: ignore[untyped-decorator]
        async def _on_state() -> None:
            self._log(f"connection state: {pc.connectionState}")

        @pc.on("iceconnectionstatechange")  # type: ignore[untyped-decorator]
        async def _on_ice_state() -> None:
            self._log(f"ICE connection state: {pc.iceConnectionState}")

        @pc.on("icegatheringstatechange")  # type: ignore[untyped-decorator]
        async def _on_ice_gathering() -> None:
            self._log(f"ICE gathering state: {pc.iceGatheringState}")

    def _log(self, message: str) -> None:
        if self._log_hook is not None:
            self._log_hook(message)

    def _add_video_track(self) -> None:
        if self._pc is None:
            raise RuntimeError("Peer connection not created.")

        video_stream_track = self._import_aiortc_symbol("VideoStreamTrack")

        class AdapterTrack(video_stream_track):  # type: ignore[misc, valid-type]
            def __init__(self, adapter: _AdapterVideoTrack, sender: VideoWebRTCSender) -> None:
                super().__init__()
                self._adapter = adapter
                self._sender = sender

            async def recv(self) -> Any:
                try:
                    frame = await self._adapter.recv()
                except Exception as exc:
                    import traceback

                    self._sender._log(
                        f"recv() error: {exc}\n{''.join(traceback.format_exc())}"
                    )
                    raise
                self._sender._frames_sent += 1
                if frame is None:
                    self._sender._frame_drops += 1
                return frame

        video_format = self._source.get_format()
        track = AdapterTrack(_AdapterVideoTrack(self._source, fps=video_format.fps), self)
        self._pc.addTrack(track)

    def _wire_ice_callbacks(self) -> None:
        if self._pc is None or self._on_local_ice_candidate is None:
            return

        @self._pc.on("icecandidate")  # type: ignore[untyped-decorator]
        async def _on_icecandidate(candidate: Any) -> None:
            if candidate is None:
                return
            candidate_str = str(getattr(candidate, "candidate", ""))
            self._log(f"local ICE candidate: {candidate_str[:80]}")
            payload = {
                "candidate": candidate_str,
                "sdpMid": getattr(candidate, "sdpMid", None),
                "sdpMLineIndex": getattr(candidate, "sdpMLineIndex", None),
            }
            assert self._on_local_ice_candidate is not None
            await self._on_local_ice_candidate(payload)

    def _force_h264_codec_if_possible(self) -> None:
        """Prefer H.264 on send transceiver when runtime supports codec controls."""
        if self._pc is None:
            return
        try:
            rtc_rtp_sender = self._import_aiortc_symbol("RTCRtpSender")
            capabilities = rtc_rtp_sender.getCapabilities("video")
            codecs = [
                codec
                for codec in getattr(capabilities, "codecs", [])
                if str(getattr(codec, "mimeType", "")).lower() == "video/h264"
            ]
            if not codecs:
                return
            for transceiver in self._pc.getTransceivers():
                if getattr(transceiver, "kind", "") == "video":
                    transceiver.setCodecPreferences(codecs)
                    self._h264_forced = True
        except Exception:
            return

    def _import_aiortc_symbol(self, symbol: str) -> Any:
        try:
            module = __import__("aiortc", fromlist=[symbol])
            return getattr(module, symbol)
        except Exception as exc:
            raise RuntimeError(
                "aiortc is required for video streaming. "
                "Install with: pip install hand-tracking-sdk[video]"
            ) from exc

    def _import_aiortc_sdp_symbol(self, symbol: str) -> Any:
        try:
            module = __import__("aiortc.sdp", fromlist=[symbol])
            return getattr(module, symbol)
        except Exception as exc:
            raise RuntimeError("aiortc.sdp helpers unavailable.") from exc
