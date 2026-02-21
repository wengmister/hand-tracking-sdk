"""Signaling schema models for host/Quest video session control."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class SignalingProtocolError(ValueError):
    """Raised when a signaling message is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class SignalingMessage:
    """Typed signaling envelope used on the WebSocket control channel."""

    type: str
    session_id: str
    payload: dict[str, Any]

    def to_json(self) -> str:
        """Serialize envelope to compact JSON."""
        return json.dumps(
            {"type": self.type, "session_id": self.session_id, "payload": self.payload},
            separators=(",", ":"),
        )


def parse_signaling_message(raw: str) -> SignalingMessage:
    """Parse one JSON signaling envelope."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SignalingProtocolError("Invalid signaling JSON.") from exc

    if not isinstance(obj, dict):
        raise SignalingProtocolError("Signaling message must be an object.")

    type_value = obj.get("type")
    session_id = obj.get("session_id")
    payload = obj.get("payload", {})
    if not isinstance(type_value, str) or not type_value:
        raise SignalingProtocolError("Message field 'type' must be a non-empty string.")
    if not isinstance(session_id, str) or not session_id:
        raise SignalingProtocolError("Message field 'session_id' must be a non-empty string.")
    if not isinstance(payload, dict):
        raise SignalingProtocolError("Message field 'payload' must be an object.")

    return SignalingMessage(type=type_value, session_id=session_id, payload=payload)


def make_signaling_message(
    *,
    type: str,
    session_id: str,
    payload: dict[str, Any] | None = None,
) -> SignalingMessage:
    """Create a signaling envelope with normalized payload."""
    return SignalingMessage(type=type, session_id=session_id, payload=payload or {})
