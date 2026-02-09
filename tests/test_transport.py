from __future__ import annotations

import socket
import threading
import time

import pytest

from hand_tracking_sdk import (
    TCPClientConfig,
    TCPClientLineReceiver,
    TCPServerConfig,
    TCPServerLineReceiver,
    TransportDisconnectedError,
    TransportTimeoutError,
    UDPLineReceiver,
    UDPReceiverConfig,
)


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def test_udp_receiver_reads_line() -> None:
    receiver = UDPLineReceiver(UDPReceiverConfig(host="127.0.0.1", port=0, timeout_s=0.2))
    with receiver:
        _, port = receiver.local_address

        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto(b"Right wrist:, 0, 0, 0, 0, 0, 0, 1\n", ("127.0.0.1", port))
        sender.close()

        line = receiver.recv_line()

    assert line == "Right wrist:, 0, 0, 0, 0, 0, 0, 1"


def test_udp_receiver_timeout() -> None:
    receiver = UDPLineReceiver(UDPReceiverConfig(host="127.0.0.1", port=0, timeout_s=0.05))
    with receiver:
        with pytest.raises(TransportTimeoutError):
            receiver.recv_line()


def test_tcp_server_receiver_reads_and_recovers_disconnect() -> None:
    receiver = TCPServerLineReceiver(
        TCPServerConfig(host="127.0.0.1", port=0, accept_timeout_s=0.2, read_timeout_s=0.2)
    )

    with receiver:
        _, port = receiver.local_address

        sender_one = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender_one.connect(("127.0.0.1", port))
        sender_one.sendall(b"line-1\nline-2\n")
        sender_one.close()

        assert receiver.recv_line() == "line-1"
        assert receiver.recv_line() == "line-2"

        with pytest.raises(TransportDisconnectedError):
            receiver.recv_line()

        sender_two = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender_two.connect(("127.0.0.1", port))
        sender_two.sendall(b"line-3\n")
        sender_two.close()

        assert receiver.recv_line() == "line-3"


def test_tcp_client_receiver_reads_line() -> None:
    port = _find_free_port()
    server_ready = threading.Event()

    def _server() -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", port))
        server.listen(1)
        server_ready.set()
        conn, _ = server.accept()
        conn.sendall(b"Left wrist:, 1, 2, 3, 4, 5, 6, 7\n")
        conn.close()
        server.close()

    thread = threading.Thread(target=_server, daemon=True)
    thread.start()
    server_ready.wait(timeout=1.0)

    receiver = TCPClientLineReceiver(
        TCPClientConfig(host="127.0.0.1", port=port, connect_timeout_s=0.5, read_timeout_s=0.5)
    )
    with receiver:
        line = receiver.recv_line()

    assert line == "Left wrist:, 1, 2, 3, 4, 5, 6, 7"


def test_tcp_server_accept_timeout() -> None:
    receiver = TCPServerLineReceiver(
        TCPServerConfig(host="127.0.0.1", port=0, accept_timeout_s=0.05, read_timeout_s=0.05)
    )

    with receiver:
        with pytest.raises(TransportTimeoutError):
            receiver.recv_line()


def test_tcp_client_reconnect_iter_lines() -> None:
    port = _find_free_port()
    server_ready = threading.Event()

    def _server() -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", port))
        server.listen(2)
        server_ready.set()

        conn1, _ = server.accept()
        conn1.sendall(b"a\n")
        conn1.close()

        time.sleep(0.1)

        conn2, _ = server.accept()
        conn2.sendall(b"b\n")
        conn2.close()

        server.close()

    thread = threading.Thread(target=_server, daemon=True)
    thread.start()
    server_ready.wait(timeout=1.0)

    receiver = TCPClientLineReceiver(
        TCPClientConfig(
            host="127.0.0.1",
            port=port,
            connect_timeout_s=0.5,
            read_timeout_s=0.2,
            reconnect_delay_s=0.02,
        )
    )

    lines: list[str] = []
    iterator = receiver.iter_lines()
    try:
        lines.append(next(iterator))
        lines.append(next(iterator))
    finally:
        receiver.close()

    assert lines == ["a", "b"]
