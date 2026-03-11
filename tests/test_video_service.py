from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from hand_tracking_sdk.video.schemas import SignalingMessage
from hand_tracking_sdk.video.service import VideoService, VideoServiceConfig


@dataclass
class _FakeConnection:
    session_id: str | None = None


class _FakeSignaling:
    def __init__(self) -> None:
        self.sent: list[SignalingMessage] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send(self, _connection: Any, message: SignalingMessage) -> None:
        self.sent.append(message)


class _FakeSender:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.offer_seen: str | None = None
        self.ice_seen: list[str] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def apply_offer(self, *, sdp_offer: str) -> str:
        self.offer_seen = sdp_offer
        return "fake_answer_sdp"

    async def add_remote_ice_candidate(
        self,
        *,
        candidate: str,
        sdp_mid: str | None,
        sdp_mline_index: int | None,
    ) -> None:
        _ = sdp_mid, sdp_mline_index
        self.ice_seen.append(candidate)

    async def get_stats(self) -> Any:
        return type(
            "_Stats",
            (),
            {"fps": 30.0, "bitrate_kbps": 1200.0, "frame_drops": 0, "rtt_ms": 10.0},
        )()


def test_video_service_hello_offer_stop_flow() -> None:
    fake_sender = _FakeSender()
    service = VideoService(
        VideoServiceConfig(stats_interval_s=999.0),
        sender_factory=lambda _source, _session_id, _fps: fake_sender,  # type: ignore[return-value, arg-type]
    )
    fake_signaling = _FakeSignaling()
    service._signaling = fake_signaling  # type: ignore[assignment]

    connection = _FakeConnection()
    session_id = "sess-1"

    async def _run() -> None:
        await service._on_message(
            connection, SignalingMessage(type="hello", session_id=session_id, payload={})  # type: ignore[arg-type]
        )
        await service._on_message(
            connection, SignalingMessage(type="start_video", session_id=session_id, payload={})  # type: ignore[arg-type]
        )
        await service._on_message(
            connection,  # type: ignore[arg-type]
            SignalingMessage(
                type="offer",
                session_id=session_id,
                payload={"sdp": "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n"},
            ),
        )
        await service._on_message(
            connection,  # type: ignore[arg-type]
            SignalingMessage(
                type="ice_candidate",
                session_id=session_id,
                payload={"candidate": "cand1", "sdpMid": "0", "sdpMLineIndex": 0},
            ),
        )
        await service._on_message(
            connection, SignalingMessage(type="stop_video", session_id=session_id, payload={})  # type: ignore[arg-type]
        )

    asyncio.run(_run())

    sent_types = [message.type for message in fake_signaling.sent]
    assert "hello_ack" in sent_types
    assert "answer" in sent_types
    assert "video_state" in sent_types
    assert fake_sender.started is True
    assert fake_sender.stopped is True
    assert fake_sender.offer_seen == "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
    assert fake_sender.ice_seen == ["cand1"]
