"""Socket transport primitives for ingesting HTS text lines."""

from __future__ import annotations

import socket
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from select import select
from time import sleep
from typing import cast

from hand_tracking_sdk.exceptions import (
    TransportClosedError,
    TransportDisconnectedError,
    TransportTimeoutError,
)


@dataclass(frozen=True, slots=True)
class UDPReceiverConfig:
    """Configuration for UDP line reception.

    :param host:
        Local bind address.
    :param port:
        Local bind port. Use ``0`` for OS-assigned ephemeral port.
    :param timeout_s:
        Receive timeout in seconds.
    :param max_datagram_size:
        Maximum datagram size in bytes for ``recvfrom``.
    :param encoding:
        Encoding used to decode bytes into text.
    """

    host: str = "0.0.0.0"
    port: int = 9000
    timeout_s: float = 1.0
    max_datagram_size: int = 65_535
    encoding: str = "utf-8"


@dataclass(frozen=True, slots=True)
class TCPServerConfig:
    """Configuration for TCP server line reception.

    :param host:
        Local bind address for incoming HTS TCP connections.
    :param port:
        Local bind port.
    :param accept_timeout_s:
        Timeout for waiting on a new client connection.
    :param read_timeout_s:
        Timeout while waiting for bytes from connected clients.
    :param backlog:
        Listen backlog used in ``socket.listen``.
    :param max_line_bytes:
        Upper bound for a buffered text line.
    :param encoding:
        Encoding used to decode bytes into text.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    accept_timeout_s: float = 1.0
    read_timeout_s: float = 1.0
    backlog: int = 5
    max_line_bytes: int = 262_144
    encoding: str = "utf-8"


@dataclass(frozen=True, slots=True)
class TCPClientConfig:
    """Configuration for TCP client line reception.

    :param host:
        Remote server address.
    :param port:
        Remote server port.
    :param connect_timeout_s:
        Timeout when establishing TCP connection.
    :param read_timeout_s:
        Timeout while waiting for bytes from the server.
    :param reconnect_delay_s:
        Delay between reconnect attempts in :meth:`TCPClientLineReceiver.iter_lines`.
    :param max_line_bytes:
        Upper bound for a buffered text line.
    :param encoding:
        Encoding used to decode bytes into text.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 1.0
    reconnect_delay_s: float = 0.25
    max_line_bytes: int = 262_144
    encoding: str = "utf-8"


class UDPLineReceiver:
    """Receive UTF-8 HTS lines over UDP datagrams."""

    def __init__(self, config: UDPReceiverConfig | None = None) -> None:
        """Create a UDP receiver.

        :param config:
            Optional receiver configuration. Defaults to :class:`UDPReceiverConfig`.
        """
        self._config = config or UDPReceiverConfig()
        self._socket: socket.socket | None = None
        self._pending_lines: deque[str] = deque()

    @property
    def local_address(self) -> tuple[str, int]:
        """Return currently bound local ``(host, port)``.

        :raises TransportClosedError:
            If the receiver is not open.
        """
        if self._socket is None:
            raise TransportClosedError("UDP receiver is not open.")
        return cast(tuple[str, int], self._socket.getsockname())

    def open(self) -> None:
        """Open and bind the UDP socket.

        :raises RuntimeError:
            If called while already open.
        """
        if self._socket is not None:
            raise RuntimeError("UDP receiver is already open.")

        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.settimeout(self._config.timeout_s)
        udp_socket.bind((self._config.host, self._config.port))
        self._socket = udp_socket

    def close(self) -> None:
        """Close the UDP socket if open."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._pending_lines.clear()

    def __enter__(self) -> UDPLineReceiver:
        """Open receiver when entering context manager."""
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """Close receiver when leaving context manager."""
        self.close()

    def recv_line(self) -> str:
        """Receive the next decoded non-empty line from UDP datagrams.

        :returns:
            Decoded line with surrounding whitespace removed.
        :raises TransportClosedError:
            If receiver socket is not open.
        :raises TransportTimeoutError:
            If no new datagram arrives before timeout and no queued lines remain.
        """
        if self._socket is None:
            raise TransportClosedError("UDP receiver is not open.")

        if self._pending_lines:
            return self._pending_lines.popleft()

        while True:
            try:
                payload, _ = self._socket.recvfrom(self._config.max_datagram_size)
            except TimeoutError as exc:
                raise TransportTimeoutError("Timed out waiting for UDP packet.") from exc

            text = payload.decode(self._config.encoding, errors="strict")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                continue

            self._pending_lines.extend(lines[1:])
            return lines[0]

    def iter_lines(self) -> Iterator[str]:
        """Yield lines until receiver is closed.

        Timeout events are ignored so callers can stop by calling :meth:`close`.

        :returns:
            Iterator of decoded text lines.
        """
        while self._socket is not None:
            try:
                yield self.recv_line()
            except TransportTimeoutError:
                continue


class TCPServerLineReceiver:
    """Receive UTF-8 HTS lines from one or more inbound TCP client connections."""

    def __init__(self, config: TCPServerConfig | None = None) -> None:
        """Create a TCP server receiver.

        :param config:
            Optional receiver configuration. Defaults to :class:`TCPServerConfig`.
        """
        self._config = config or TCPServerConfig()
        self._server_socket: socket.socket | None = None
        self._client_sockets: set[socket.socket] = set()
        self._client_buffers: dict[socket.socket, bytearray] = {}
        self._pending_lines: deque[str] = deque()

    @property
    def local_address(self) -> tuple[str, int]:
        """Return currently bound local ``(host, port)``.

        :raises TransportClosedError:
            If the server socket is not open.
        """
        if self._server_socket is None:
            raise TransportClosedError("TCP server receiver is not open.")
        return cast(tuple[str, int], self._server_socket.getsockname())

    def open(self) -> None:
        """Open and bind listening TCP socket.

        :raises RuntimeError:
            If called while already open.
        """
        if self._server_socket is not None:
            raise RuntimeError("TCP server receiver is already open.")

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self._config.host, self._config.port))
        server_socket.listen(self._config.backlog)
        server_socket.setblocking(False)
        self._server_socket = server_socket

    def close(self) -> None:
        """Close connected client and server sockets."""
        for client_socket in list(self._client_sockets):
            self._close_client_socket(client_socket)

        if self._server_socket is not None:
            self._server_socket.close()
            self._server_socket = None

        self._pending_lines.clear()

    def __enter__(self) -> TCPServerLineReceiver:
        """Open receiver when entering context manager."""
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """Close receiver when leaving context manager."""
        self.close()

    def _accept_ready_clients(self) -> None:
        """Accept all currently pending inbound TCP client connections.

        :raises TransportClosedError:
            If the server socket is not open.
        """
        if self._server_socket is None:
            raise TransportClosedError("TCP server receiver is not open.")

        while True:
            try:
                client_socket, _ = self._server_socket.accept()
            except BlockingIOError:
                break
            client_socket.setblocking(False)
            self._client_sockets.add(client_socket)
            self._client_buffers[client_socket] = bytearray()

    def _close_client_socket(self, client_socket: socket.socket) -> None:
        """Close one connected TCP client and clear its buffer."""
        try:
            client_socket.close()
        finally:
            self._client_sockets.discard(client_socket)
            self._client_buffers.pop(client_socket, None)

    def _drain_ready_client(self, client_socket: socket.socket) -> bool:
        """Read bytes from one ready client and enqueue complete text lines.

        :returns:
            ``True`` if the client disconnected or was dropped, ``False`` otherwise.
        """
        buffer = self._client_buffers.get(client_socket)
        if buffer is None:
            return True

        try:
            chunk = client_socket.recv(4096)
        except BlockingIOError:
            return False
        except OSError:
            self._close_client_socket(client_socket)
            return True

        if not chunk:
            self._close_client_socket(client_socket)
            return True

        buffer.extend(chunk)
        if len(buffer) >= self._config.max_line_bytes:
            self._close_client_socket(client_socket)
            return True

        while True:
            newline_index = buffer.find(b"\n")
            if newline_index < 0:
                break
            raw = bytes(buffer[:newline_index])
            del buffer[: newline_index + 1]
            self._pending_lines.append(raw.decode(self._config.encoding, errors="strict").strip())

        return False

    def recv_line(self) -> str:
        """Receive one newline-terminated line from any connected client.

        If no client is connected, this method waits for at least one inbound
        connection before reading lines.

        :returns:
            Decoded line with trailing newline removed.
        :raises TransportClosedError:
            If the server receiver is not open.
        :raises TransportTimeoutError:
            If waiting for connection or data times out.
        :raises TransportDisconnectedError:
            If all connected clients disconnect before a new line is received.
        """
        if self._server_socket is None:
            raise TransportClosedError("TCP server receiver is not open.")

        if self._pending_lines:
            return self._pending_lines.popleft()

        while True:
            wait_for_connection = not self._client_sockets
            timeout_s = (
                self._config.accept_timeout_s
                if wait_for_connection
                else self._config.read_timeout_s
            )
            ready_sockets, _, _ = select(
                [self._server_socket, *self._client_sockets],
                [],
                [],
                timeout_s,
            )
            if not ready_sockets:
                if wait_for_connection:
                    raise TransportTimeoutError("Timed out waiting for TCP client connection.")
                raise TransportTimeoutError("Timed out waiting for TCP data.")

            disconnected = False
            if self._server_socket in ready_sockets:
                self._accept_ready_clients()

            for client_socket in list(self._client_sockets):
                if client_socket in ready_sockets:
                    disconnected = self._drain_ready_client(client_socket) or disconnected

            if self._pending_lines:
                return self._pending_lines.popleft()

            if disconnected and not self._client_sockets:
                raise TransportDisconnectedError("TCP client disconnected.")

    def iter_lines(self) -> Iterator[str]:
        """Yield lines continuously, recovering from disconnects.

        Timeout and disconnect events are treated as transient.

        :returns:
            Iterator of decoded text lines.
        """
        while self._server_socket is not None:
            try:
                yield self.recv_line()
            except (TransportTimeoutError, TransportDisconnectedError):
                continue


class TCPClientLineReceiver:
    """Receive UTF-8 HTS lines from an outbound TCP client connection."""

    def __init__(self, config: TCPClientConfig) -> None:
        """Create a TCP client receiver.

        :param config:
            Client connection and read settings.
        """
        self._config = config
        self._socket: socket.socket | None = None
        self._buffer = bytearray()

    def open(self) -> None:
        """Open TCP connection to configured host and port.

        :raises RuntimeError:
            If called while already open.
        :raises OSError:
            If the remote endpoint cannot be reached.
        """
        if self._socket is not None:
            raise RuntimeError("TCP client receiver is already open.")

        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(self._config.connect_timeout_s)
        tcp_socket.connect((self._config.host, self._config.port))
        tcp_socket.settimeout(self._config.read_timeout_s)
        self._socket = tcp_socket

    def close(self) -> None:
        """Close TCP client socket if open."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._buffer.clear()

    def __enter__(self) -> TCPClientLineReceiver:
        """Open receiver when entering context manager."""
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """Close receiver when leaving context manager."""
        self.close()

    def recv_line(self) -> str:
        """Receive one newline-terminated line from TCP stream.

        :returns:
            Decoded line with trailing newline removed.
        :raises TransportClosedError:
            If socket is not open.
        :raises TransportTimeoutError:
            If waiting for data times out.
        :raises TransportDisconnectedError:
            If remote endpoint disconnects.
        """
        if self._socket is None:
            raise TransportClosedError("TCP client receiver is not open.")

        while True:
            newline_index = self._buffer.find(b"\n")
            if newline_index >= 0:
                raw = bytes(self._buffer[:newline_index])
                del self._buffer[: newline_index + 1]
                return raw.decode(self._config.encoding, errors="strict").strip()

            if len(self._buffer) >= self._config.max_line_bytes:
                self._buffer.clear()
                raise TransportDisconnectedError("Buffered TCP line exceeded max size.")

            try:
                chunk = self._socket.recv(4096)
            except TimeoutError as exc:
                raise TransportTimeoutError("Timed out waiting for TCP data.") from exc

            if not chunk:
                self.close()
                raise TransportDisconnectedError("TCP server disconnected.")

            self._buffer.extend(chunk)

    def iter_lines(self) -> Iterator[str]:
        """Yield lines continuously and reconnect on disconnect.

        :returns:
            Iterator of decoded text lines.
        """
        while True:
            if self._socket is None:
                self.open()

            try:
                yield self.recv_line()
            except TransportTimeoutError:
                continue
            except TransportDisconnectedError:
                sleep(self._config.reconnect_delay_s)
                continue
