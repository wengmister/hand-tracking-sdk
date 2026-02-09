"""Frame assembly utilities for combining wrist and landmark packets."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic_ns, time_ns

from hand_tracking_sdk.models import (
    HandLandmarks,
    HandSide,
    LandmarksPacket,
    ParsedPacket,
    WristPacket,
    WristPose,
)
from hand_tracking_sdk.parser import parse_line


@dataclass(frozen=True, slots=True)
class HandFrame:
    """Coherent per-hand frame assembled from wrist and landmark packets.

    :param side:
        Hand side for this frame.
    :param wrist:
        Wrist pose payload.
    :param landmarks:
        Ordered set of 21 hand landmarks.
    :param sequence_id:
        Monotonic sequence number per hand side, incremented on each emitted frame.
    :param recv_ts_ns:
        Monotonic receive timestamp for the assembled frame.
    :param recv_time_unix_ns:
        Optional wall-clock timestamp in Unix nanoseconds.
    :param source_ts_ns:
        Optional source timestamp supplied by upstream sender.
    :param wrist_recv_ts_ns:
        Receive timestamp of the wrist payload included in this frame.
    :param landmarks_recv_ts_ns:
        Receive timestamp of the landmark payload included in this frame.
    """

    side: HandSide
    wrist: WristPose
    landmarks: HandLandmarks
    sequence_id: int
    recv_ts_ns: int
    recv_time_unix_ns: int | None
    source_ts_ns: int | None
    wrist_recv_ts_ns: int
    landmarks_recv_ts_ns: int


@dataclass(slots=True)
class _SideAssemblyState:
    wrist: WristPose | None = None
    wrist_recv_ts_ns: int | None = None
    landmarks: HandLandmarks | None = None
    landmarks_recv_ts_ns: int | None = None
    last_emitted_wrist_recv_ts_ns: int | None = None
    last_emitted_landmarks_recv_ts_ns: int | None = None
    next_sequence_id: int = 0


class HandFrameAssembler:
    """Assemble coherent frames from incoming parsed HTS packets.

    Emission policy:
    - A frame is emitted only after both wrist and landmarks are available for a hand side.
    - A new frame is emitted when at least one component timestamp advances.
    - Stale component updates (older timestamps than currently stored) are ignored.
    """

    def __init__(self, *, include_wall_time: bool = True) -> None:
        """Create a frame assembler.

        :param include_wall_time:
            If ``True``, :class:`HandFrame` includes ``recv_time_unix_ns`` using
            ``time.time_ns()`` when caller does not provide one.
        """
        self._include_wall_time = include_wall_time
        self._state: dict[HandSide, _SideAssemblyState] = {
            HandSide.LEFT: _SideAssemblyState(),
            HandSide.RIGHT: _SideAssemblyState(),
        }

    def reset(self, side: HandSide | None = None) -> None:
        """Reset assembler state.

        :param side:
            Optional side to reset. If omitted, both sides are reset.
        """
        if side is None:
            self._state[HandSide.LEFT] = _SideAssemblyState()
            self._state[HandSide.RIGHT] = _SideAssemblyState()
            return

        self._state[side] = _SideAssemblyState()

    def push_packet(
        self,
        packet: ParsedPacket,
        *,
        recv_ts_ns: int | None = None,
        recv_time_unix_ns: int | None = None,
        source_ts_ns: int | None = None,
    ) -> HandFrame | None:
        """Push one parsed packet and optionally emit a coherent frame.

        :param packet:
            Parsed wrist or landmarks packet.
        :param recv_ts_ns:
            Monotonic receive timestamp in nanoseconds. If omitted, generated
            with ``time.monotonic_ns()``.
        :param recv_time_unix_ns:
            Optional Unix wall-clock timestamp in nanoseconds. If omitted and
            ``include_wall_time=True``, generated with ``time.time_ns()``.
        :param source_ts_ns:
            Optional source timestamp supplied by upstream sender.
        :returns:
            A newly assembled frame or ``None`` if frame is incomplete/unchanged.
        """
        recv_ts_ns_value, recv_time_unix_ns_value = self._resolve_timestamps(
            recv_ts_ns=recv_ts_ns,
            recv_time_unix_ns=recv_time_unix_ns,
        )

        side_state = self._state[packet.side]
        if isinstance(packet, WristPacket):
            if (
                side_state.wrist_recv_ts_ns is not None
                and recv_ts_ns_value < side_state.wrist_recv_ts_ns
            ):
                return None
            side_state.wrist = packet.data
            side_state.wrist_recv_ts_ns = recv_ts_ns_value
        elif isinstance(packet, LandmarksPacket):
            if (
                side_state.landmarks_recv_ts_ns is not None
                and recv_ts_ns_value < side_state.landmarks_recv_ts_ns
            ):
                return None
            side_state.landmarks = packet.data
            side_state.landmarks_recv_ts_ns = recv_ts_ns_value

        return self._maybe_emit_frame(
            side=packet.side,
            recv_time_unix_ns=recv_time_unix_ns_value,
            source_ts_ns=source_ts_ns,
        )

    def push_line(
        self,
        line: str,
        *,
        recv_ts_ns: int | None = None,
        recv_time_unix_ns: int | None = None,
        source_ts_ns: int | None = None,
    ) -> HandFrame | None:
        """Parse and push one raw HTS line into assembler state.

        :param line:
            Raw UTF-8 decoded HTS CSV line.
        :param recv_ts_ns:
            Monotonic receive timestamp in nanoseconds.
        :param recv_time_unix_ns:
            Optional Unix wall-clock timestamp in nanoseconds.
        :param source_ts_ns:
            Optional source timestamp supplied by upstream sender.
        :returns:
            A newly assembled frame or ``None`` if frame is incomplete/unchanged.
        """
        packet = parse_line(line)
        return self.push_packet(
            packet,
            recv_ts_ns=recv_ts_ns,
            recv_time_unix_ns=recv_time_unix_ns,
            source_ts_ns=source_ts_ns,
        )

    def _resolve_timestamps(
        self,
        *,
        recv_ts_ns: int | None,
        recv_time_unix_ns: int | None,
    ) -> tuple[int, int | None]:
        recv_ts_ns_value = monotonic_ns() if recv_ts_ns is None else recv_ts_ns
        if recv_time_unix_ns is not None:
            return recv_ts_ns_value, recv_time_unix_ns

        if self._include_wall_time:
            return recv_ts_ns_value, time_ns()
        return recv_ts_ns_value, None

    def _maybe_emit_frame(
        self,
        *,
        side: HandSide,
        recv_time_unix_ns: int | None,
        source_ts_ns: int | None,
    ) -> HandFrame | None:
        side_state = self._state[side]
        if (
            side_state.wrist is None
            or side_state.wrist_recv_ts_ns is None
            or side_state.landmarks is None
            or side_state.landmarks_recv_ts_ns is None
        ):
            return None

        has_new_wrist = side_state.last_emitted_wrist_recv_ts_ns != side_state.wrist_recv_ts_ns
        has_new_landmarks = (
            side_state.last_emitted_landmarks_recv_ts_ns != side_state.landmarks_recv_ts_ns
        )
        if not has_new_wrist and not has_new_landmarks:
            return None

        sequence_id = side_state.next_sequence_id
        side_state.next_sequence_id += 1

        side_state.last_emitted_wrist_recv_ts_ns = side_state.wrist_recv_ts_ns
        side_state.last_emitted_landmarks_recv_ts_ns = side_state.landmarks_recv_ts_ns

        return HandFrame(
            side=side,
            wrist=side_state.wrist,
            landmarks=side_state.landmarks,
            sequence_id=sequence_id,
            recv_ts_ns=max(side_state.wrist_recv_ts_ns, side_state.landmarks_recv_ts_ns),
            recv_time_unix_ns=recv_time_unix_ns,
            source_ts_ns=source_ts_ns,
            wrist_recv_ts_ns=side_state.wrist_recv_ts_ns,
            landmarks_recv_ts_ns=side_state.landmarks_recv_ts_ns,
        )
