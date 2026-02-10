"""High-level streaming client API for HTS telemetry."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from hand_tracking_sdk.exceptions import ParseError
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

    def iter_events(self) -> Iterator[StreamEvent]:
        """Iterate streaming events from configured transport.

        :returns:
            Iterator yielding packet and/or frame events per configured output mode.
        :raises ParseError:
            When `error_policy=strict` and an incoming line cannot be parsed.
        """
        receiver = self._make_receiver()
        with receiver:
            for line in receiver.iter_lines():
                packet = self._parse_with_policy(line)
                if packet is None:
                    continue
                if not self._matches_hand_filter(packet.side):
                    continue

                if self._config.output in (StreamOutput.PACKETS, StreamOutput.BOTH):
                    yield packet

                if self._config.output in (StreamOutput.FRAMES, StreamOutput.BOTH):
                    frame = self._frame_assembler.push_packet(packet)
                    if frame is not None:
                        yield frame

    def run(
        self,
        callback: Callable[[StreamEvent], None],
        *,
        max_events: int | None = None,
    ) -> int:
        """Run stream loop and invoke callback for each emitted event.

        :param callback:
            Function called for each emitted stream event.
        :param max_events:
            Optional cap on processed events; useful for controlled execution.
        :returns:
            Number of callback invocations performed.
        :raises ParseError:
            When `error_policy=strict` and parsing fails.
        """
        processed = 0
        for event in self.iter_events():
            callback(event)
            processed += 1
            if max_events is not None and processed >= max_events:
                return processed
        return processed

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
        except ParseError:
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
