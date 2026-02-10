"""High-level streaming client API for HTS telemetry."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from hand_tracking_sdk.exceptions import (
    ClientCallbackError,
    ClientConfigurationError,
    ParseError,
)
from hand_tracking_sdk.frame import HandFrame, HandFrameAssembler
from hand_tracking_sdk.models import HandSide, ParsedPacket
from hand_tracking_sdk.parser import parse_line
from hand_tracking_sdk.transport import (
    TCPClientConfig,
    TCPClientLineReceiver,
    TCPServerConfig,
    TCPServerLineReceiver,
    UDPLineReceiver,
    UDPReceiverConfig,
)


class TransportMode(StrEnum):
    """Transport mode used by :class:`HTSClient`."""

    UDP = "udp"
    TCP_SERVER = "tcp_server"
    TCP_CLIENT = "tcp_client"


class StreamOutput(StrEnum):
    """Output type emitted by :meth:`HTSClient.iter_events`."""

    PACKETS = "packets"
    FRAMES = "frames"
    BOTH = "both"


class HandFilter(StrEnum):
    """Hand-side filter applied before packet/frame emission."""

    BOTH = "both"
    LEFT = "left"
    RIGHT = "right"


class ErrorPolicy(StrEnum):
    """Error policy applied to parse failures."""

    STRICT = "strict"
    TOLERANT = "tolerant"


class LogEventKind(StrEnum):
    """Structured log event kinds emitted by :class:`HTSClient`."""

    RECEIVED_LINE = "received_line"
    PARSE_ERROR = "parse_error"
    FILTERED_PACKET = "filtered_packet"
    EMITTED_PACKET = "emitted_packet"
    EMITTED_FRAME = "emitted_frame"
    CALLBACK_ERROR = "callback_error"


@dataclass(frozen=True, slots=True)
class StreamLogEvent:
    """Structured client log event for observability hooks.

    :param kind:
        Event kind discriminator.
    :param message:
        Human-readable event message.
    :param side:
        Optional hand side associated with the event.
    :param line:
        Optional raw input line associated with the event.
    :param exception:
        Optional exception associated with the event.
    """

    kind: LogEventKind
    message: str
    side: HandSide | None = None
    line: str | None = None
    exception: Exception | None = None


@dataclass(frozen=True, slots=True)
class ClientStats:
    """Observable counters for :class:`HTSClient` runtime behavior."""

    lines_received: int = 0
    parse_errors: int = 0
    dropped_lines: int = 0
    packets_filtered: int = 0
    packets_emitted: int = 0
    frames_emitted: int = 0
    callbacks_invoked: int = 0
    callback_errors: int = 0


@dataclass(frozen=True, slots=True)
class HTSClientConfig:
    """Configuration for high-level HTS streaming client.

    :param transport_mode:
        Network transport mode.
    :param host:
        Host address used for bind/connect according to transport mode.
    :param port:
        Port used for bind/connect according to transport mode.
    :param timeout_s:
        I/O timeout in seconds for receive operations. In ``tcp_server`` mode,
        initial connection wait uses ``max(timeout_s, 5.0)`` to avoid premature
        startup timeouts while waiting for a device to connect.
    :param reconnect_delay_s:
        Delay used by TCP client reconnect loop.
    :param output:
        Event output mode (`packets`, `frames`, or `both`).
    :param hand_filter:
        Hand-side filter for emitted events.
    :param error_policy:
        Parse error handling strategy.
    :param include_wall_time:
        Whether assembled frames include `recv_time_unix_ns` by default.
    :param log_hook:
        Optional structured log callback invoked for client lifecycle events.
    """

    transport_mode: TransportMode = TransportMode.UDP
    host: str = "0.0.0.0"
    port: int = 9000
    timeout_s: float = 1.0
    reconnect_delay_s: float = 0.25
    output: StreamOutput = StreamOutput.FRAMES
    hand_filter: HandFilter = HandFilter.BOTH
    error_policy: ErrorPolicy = ErrorPolicy.STRICT
    include_wall_time: bool = True
    log_hook: Callable[[StreamLogEvent], None] | None = None

    def __post_init__(self) -> None:
        """Validate configuration constraints.

        :raises ClientConfigurationError:
            If one or more fields are invalid for runtime operation.
        """
        if not self.host:
            raise ClientConfigurationError("host must not be empty.")
        if self.port < 0 or self.port > 65535:
            raise ClientConfigurationError("port must be in range [0, 65535].")
        if self.timeout_s <= 0:
            raise ClientConfigurationError("timeout_s must be greater than 0.")
        if self.reconnect_delay_s < 0:
            raise ClientConfigurationError("reconnect_delay_s must be non-negative.")


class _LineReceiver(Protocol):
    """Protocol for line-oriented transport receivers used by :class:`HTSClient`."""

    def __enter__(self) -> _LineReceiver: ...

    def __exit__(self, *_: object) -> None: ...

    def iter_lines(self) -> Iterator[str]: ...


StreamEvent = ParsedPacket | HandFrame
"""Public event type emitted by :class:`HTSClient` streaming methods."""


class HTSClient:
    """High-level client for streaming parsed packets and assembled frames."""

    def __init__(
        self,
        config: HTSClientConfig,
        *,
        receiver_factory: Callable[[HTSClientConfig], _LineReceiver] | None = None,
    ) -> None:
        """Create a streaming client.

        :param config:
            Client configuration.
        :param receiver_factory:
            Optional factory for line receiver dependency injection.
        """
        self._config = config
        self._receiver_factory = receiver_factory
        self._frame_assembler = HandFrameAssembler(include_wall_time=config.include_wall_time)
        self._stats = ClientStats()

    def iter_events(self) -> Iterator[StreamEvent]:
        """Iterate streaming events from configured transport.

        :returns:
            Iterator yielding packet and/or frame events per configured output mode.
        :raises ParseError:
            When ``error_policy=strict`` and an incoming line cannot be parsed.
        """
        receiver = self._make_receiver()
        with receiver:
            for line in receiver.iter_lines():
                self._stats = self._stats_with(lines_received=self._stats.lines_received + 1)
                self._emit_log(
                    StreamLogEvent(
                        kind=LogEventKind.RECEIVED_LINE,
                        message="Received input line.",
                        line=line,
                    )
                )

                packet = self._parse_with_policy(line)
                if packet is None:
                    continue

                if not self._matches_hand_filter(packet.side):
                    self._stats = self._stats_with(
                        packets_filtered=self._stats.packets_filtered + 1,
                        dropped_lines=self._stats.dropped_lines + 1,
                    )
                    self._emit_log(
                        StreamLogEvent(
                            kind=LogEventKind.FILTERED_PACKET,
                            message="Packet dropped due to hand filter.",
                            side=packet.side,
                        )
                    )
                    continue

                if self._config.output in (StreamOutput.PACKETS, StreamOutput.BOTH):
                    self._stats = self._stats_with(packets_emitted=self._stats.packets_emitted + 1)
                    self._emit_log(
                        StreamLogEvent(
                            kind=LogEventKind.EMITTED_PACKET,
                            message="Emitted packet event.",
                            side=packet.side,
                        )
                    )
                    yield packet

                if self._config.output in (StreamOutput.FRAMES, StreamOutput.BOTH):
                    frame = self._frame_assembler.push_packet(packet)
                    if frame is not None:
                        self._stats = self._stats_with(
                            frames_emitted=self._stats.frames_emitted + 1
                        )
                        self._emit_log(
                            StreamLogEvent(
                                kind=LogEventKind.EMITTED_FRAME,
                                message="Emitted frame event.",
                                side=frame.side,
                            )
                        )
                        yield frame

    def run(
        self,
        callback: Callable[[StreamEvent], None],
        *,
        max_events: int | None = None,
        wrap_callback_exceptions: bool = False,
    ) -> int:
        """Run stream loop and invoke callback for each emitted event.

        :param callback:
            Function called for each emitted stream event.
        :param max_events:
            Optional cap on processed events; useful for controlled execution.
        :param wrap_callback_exceptions:
            If ``True``, callback exceptions are re-raised as
            :class:`ClientCallbackError`.
        :returns:
            Number of callback invocations performed.
        :raises ParseError:
            When ``error_policy=strict`` and parsing fails.
        :raises ClientCallbackError:
            When callback raises and wrapping is enabled.
        """
        processed = 0
        for event in self.iter_events():
            try:
                callback(event)
            except Exception as exc:
                self._stats = self._stats_with(callback_errors=self._stats.callback_errors + 1)
                self._emit_log(
                    StreamLogEvent(
                        kind=LogEventKind.CALLBACK_ERROR,
                        message="Callback raised an exception.",
                        exception=exc,
                    )
                )
                if wrap_callback_exceptions:
                    raise ClientCallbackError("Callback failed during stream processing.") from exc
                raise

            processed += 1
            self._stats = self._stats_with(callbacks_invoked=self._stats.callbacks_invoked + 1)
            if max_events is not None and processed >= max_events:
                return processed
        return processed

    def get_stats(self) -> ClientStats:
        """Return a snapshot of current client counters."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset client counters to zero values."""
        self._stats = ClientStats()

    def _make_receiver(self) -> _LineReceiver:
        """Create a line receiver according to configured transport mode."""
        if self._receiver_factory is not None:
            return self._receiver_factory(self._config)

        if self._config.transport_mode == TransportMode.UDP:
            return UDPLineReceiver(
                UDPReceiverConfig(
                    host=self._config.host,
                    port=self._config.port,
                    timeout_s=self._config.timeout_s,
                )
            )

        if self._config.transport_mode == TransportMode.TCP_SERVER:
            return TCPServerLineReceiver(
                TCPServerConfig(
                    host=self._config.host,
                    port=self._config.port,
                    accept_timeout_s=max(self._config.timeout_s, 5.0),
                    read_timeout_s=self._config.timeout_s,
                )
            )

        return TCPClientLineReceiver(
            TCPClientConfig(
                host=self._config.host,
                port=self._config.port,
                connect_timeout_s=self._config.timeout_s,
                read_timeout_s=self._config.timeout_s,
                reconnect_delay_s=self._config.reconnect_delay_s,
            )
        )

    def _parse_with_policy(self, line: str) -> ParsedPacket | None:
        """Parse one line according to configured strict/tolerant policy."""
        try:
            return parse_line(line)
        except ParseError as exc:
            self._stats = self._stats_with(
                parse_errors=self._stats.parse_errors + 1,
                dropped_lines=self._stats.dropped_lines + 1,
            )
            self._emit_log(
                StreamLogEvent(
                    kind=LogEventKind.PARSE_ERROR,
                    message="Failed to parse input line.",
                    line=line,
                    exception=exc,
                )
            )
            if self._config.error_policy == ErrorPolicy.STRICT:
                raise
            return None

    def _matches_hand_filter(self, side: HandSide) -> bool:
        """Return whether a packet/frame side passes the configured hand filter."""
        if self._config.hand_filter == HandFilter.BOTH:
            return True
        if self._config.hand_filter == HandFilter.LEFT:
            return side == HandSide.LEFT
        return side == HandSide.RIGHT

    def _stats_with(self, **changes: int) -> ClientStats:
        """Return updated stats snapshot with selected counter changes."""
        return ClientStats(
            lines_received=changes.get("lines_received", self._stats.lines_received),
            parse_errors=changes.get("parse_errors", self._stats.parse_errors),
            dropped_lines=changes.get("dropped_lines", self._stats.dropped_lines),
            packets_filtered=changes.get("packets_filtered", self._stats.packets_filtered),
            packets_emitted=changes.get("packets_emitted", self._stats.packets_emitted),
            frames_emitted=changes.get("frames_emitted", self._stats.frames_emitted),
            callbacks_invoked=changes.get("callbacks_invoked", self._stats.callbacks_invoked),
            callback_errors=changes.get("callback_errors", self._stats.callback_errors),
        )

    def _emit_log(self, event: StreamLogEvent) -> None:
        """Emit one structured log event if a hook is configured."""
        if self._config.log_hook is not None:
            self._config.log_hook(event)
