"""WebSocket signaling server for WebRTC setup/control."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from hand_tracking_sdk.video.schemas import SignalingMessage, parse_signaling_message


@dataclass(slots=True)
class SignalingConnection:
    """One connected signaling client."""

    websocket: Any
    session_id: str | None = None


class VideoSignalingServer:
    """Async WebSocket signaling server."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        on_message: Callable[[SignalingConnection, SignalingMessage], Awaitable[None]],
        on_connect: Callable[[SignalingConnection], Awaitable[None]] | None = None,
        on_disconnect: Callable[[SignalingConnection], Awaitable[None]] | None = None,
    ) -> None:
        """Create a signaling server with message and lifecycle callbacks."""
        self._host = host
        self._port = port
        self._on_message = on_message
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._server: Any = None
        self._connections: list[SignalingConnection] = []
        self._session_map: dict[str, SignalingConnection] = {}
        self._lock = asyncio.Lock()

    @property
    def session_connections(self) -> dict[str, SignalingConnection]:
        """Return session registry snapshot."""
        return dict(self._session_map)

    async def start(self) -> None:
        """Start WebSocket server."""
        websockets = self._import_websockets()
        self._server = await websockets.serve(self._handle_client, self._host, self._port)

    async def stop(self) -> None:
        """Stop server and close all active clients."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for connection in list(self._connections):
            try:
                await connection.websocket.close()
            except Exception:
                pass

        self._connections.clear()
        self._session_map.clear()

    async def send(self, connection: SignalingConnection, message: SignalingMessage) -> None:
        """Send one signaling envelope to a connection."""
        await connection.websocket.send(message.to_json())

    async def send_to_session(self, session_id: str, message: SignalingMessage) -> bool:
        """Send one signaling envelope by session id."""
        connection = self._session_map.get(session_id)
        if connection is None:
            return False
        await self.send(connection, message)
        return True

    async def _handle_client(self, websocket: Any) -> None:
        connection = SignalingConnection(websocket=websocket)
        async with self._lock:
            self._connections.append(connection)

        try:
            if self._on_connect is not None:
                await self._on_connect(connection)
            async for raw in websocket:
                message = parse_signaling_message(str(raw))
                connection.session_id = message.session_id
                async with self._lock:
                    self._session_map[message.session_id] = connection
                await self._on_message(connection, message)
        finally:
            async with self._lock:
                if connection in self._connections:
                    self._connections.remove(connection)
                if connection.session_id is not None:
                    self._session_map.pop(connection.session_id, None)
            if self._on_disconnect is not None:
                await self._on_disconnect(connection)

    def _import_websockets(self) -> Any:
        try:
            return __import__("websockets", fromlist=["serve"])
        except Exception as exc:
            raise RuntimeError(
                "websockets is required for video signaling. "
                "Install with: pip install hand-tracking-sdk[video]"
            ) from exc
