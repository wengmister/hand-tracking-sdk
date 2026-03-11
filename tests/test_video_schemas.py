import pytest

from hand_tracking_sdk.video.schemas import (
    SignalingProtocolError,
    make_signaling_message,
    parse_signaling_message,
)


def test_parse_signaling_message_valid() -> None:
    message = parse_signaling_message(
        '{"type":"hello","session_id":"abc","payload":{"app_version":"1.0"}}'
    )
    assert message.type == "hello"
    assert message.session_id == "abc"
    assert message.payload["app_version"] == "1.0"


def test_parse_signaling_message_requires_session_id() -> None:
    with pytest.raises(SignalingProtocolError):
        parse_signaling_message('{"type":"hello","payload":{}}')


def test_make_signaling_message_serializes() -> None:
    message = make_signaling_message(type="pong", session_id="x", payload={"ts": 1})
    assert '"type":"pong"' in message.to_json()
