from __future__ import annotations

import math

from hand_tracking_sdk import (
    HandFrame,
    HandLandmarks,
    HandSide,
    WristPose,
    convert_hand_frame_unity_left_to_right,
    convert_landmarks_unity_left_to_right,
    convert_wrist_pose_unity_left_to_right,
    sdk_to_flu_position,
    unity_left_to_flu_position,
    unity_left_to_right_position,
    unity_left_to_right_quaternion,
    unity_right_to_flu_position,
)


def _quat_close(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return all(math.isclose(x, y, abs_tol=1e-6) for x, y in zip(a, b, strict=True))


def _quat_equivalent(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    # q and -q represent the same rotation.
    if _quat_close(a, b):
        return True
    return _quat_close(a, (-b[0], -b[1], -b[2], -b[3]))


def test_position_conversion_flips_y() -> None:
    assert unity_left_to_right_position(1.0, 2.0, 3.0) == (1.0, -2.0, 3.0)


def test_sdk_to_flu_position_mapping() -> None:
    assert sdk_to_flu_position(1.0, 2.0, 3.0) == (3.0, -1.0, -2.0)


def test_unity_right_to_flu_position_mapping() -> None:
    assert unity_right_to_flu_position(1.0, 2.0, 3.0) == (3.0, -1.0, -2.0)


def test_unity_left_to_flu_position_mapping() -> None:
    assert unity_left_to_flu_position(1.0, 2.0, 3.0) == (3.0, -1.0, 2.0)


def test_quaternion_identity_is_preserved() -> None:
    converted = unity_left_to_right_quaternion(0.0, 0.0, 0.0, 1.0)
    assert _quat_equivalent(converted, (0.0, 0.0, 0.0, 1.0))


def test_quaternion_x_rotation_flips_sign() -> None:
    angle = math.pi / 2.0
    source = (math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0))
    converted = unity_left_to_right_quaternion(*source)
    expected = (-math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0))
    assert _quat_equivalent(converted, expected)


def test_wrist_and_landmarks_conversion() -> None:
    pose = WristPose(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
    converted_pose = convert_wrist_pose_unity_left_to_right(pose)

    assert converted_pose.x == 1.0
    assert converted_pose.y == -2.0
    assert converted_pose.z == 3.0

    landmarks = HandLandmarks(points=((1.0, 2.0, 3.0), (4.0, -5.0, 6.0)))
    converted_landmarks = convert_landmarks_unity_left_to_right(landmarks)
    assert converted_landmarks.points == ((1.0, -2.0, 3.0), (4.0, 5.0, 6.0))


def test_hand_frame_conversion_preserves_metadata() -> None:
    frame = HandFrame(
        side=HandSide.LEFT,
        frame_id="left_hand_link",
        wrist=WristPose(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        landmarks=HandLandmarks(points=((1.0, 2.0, 3.0),)),
        sequence_id=5,
        recv_ts_ns=101,
        recv_time_unix_ns=202,
        source_ts_ns=303,
        wrist_recv_ts_ns=111,
        landmarks_recv_ts_ns=112,
    )

    converted = convert_hand_frame_unity_left_to_right(frame)

    assert converted.side == HandSide.LEFT
    assert converted.frame_id == "left_hand_link"
    assert converted.sequence_id == 5
    assert converted.recv_ts_ns == 101
    assert converted.recv_time_unix_ns == 202
    assert converted.source_ts_ns == 303
    assert converted.wrist_recv_ts_ns == 111
    assert converted.landmarks_recv_ts_ns == 112
    assert converted.wrist.y == -2.0
    assert converted.landmarks.points[0] == (1.0, -2.0, 3.0)
