from __future__ import annotations

import pytest

from hand_tracking_sdk import FingerName, HandFrame, HandLandmarks, HandSide, JointName, WristPose


def _sample_landmarks() -> HandLandmarks:
    return HandLandmarks(
        points=tuple((float(i), float(i) + 0.1, float(i) + 0.2) for i in range(21))
    )


def _sample_frame() -> HandFrame:
    return HandFrame(
        side=HandSide.RIGHT,
        frame_id="hts_right_hand",
        wrist=WristPose(x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        landmarks=_sample_landmarks(),
        sequence_id=10,
        recv_ts_ns=100,
        recv_time_unix_ns=200,
        source_ts_ns=None,
        wrist_recv_ts_ns=90,
        landmarks_recv_ts_ns=95,
    )


def test_hand_landmarks_get_joint_by_enum_and_str() -> None:
    landmarks = _sample_landmarks()

    assert landmarks.get_joint(JointName.INDEX_TIP) == (8.0, 8.1, 8.2)
    assert landmarks.get_joint("IndexTip") == (8.0, 8.1, 8.2)


def test_hand_landmarks_get_finger() -> None:
    landmarks = _sample_landmarks()
    index_joints = landmarks.get_finger(FingerName.INDEX)

    assert list(index_joints.keys()) == [
        JointName.INDEX_PROXIMAL,
        JointName.INDEX_INTERMEDIATE,
        JointName.INDEX_DISTAL,
        JointName.INDEX_TIP,
    ]
    assert index_joints[JointName.INDEX_TIP] == (8.0, 8.1, 8.2)


def test_hand_landmarks_get_wrist_group() -> None:
    landmarks = _sample_landmarks()
    wrist_group = landmarks.get_finger("wrist")

    assert wrist_group == {JointName.WRIST: (0.0, 0.1, 0.2)}


def test_hand_landmarks_invalid_joint_and_finger() -> None:
    landmarks = _sample_landmarks()

    with pytest.raises(ValueError, match="Unknown joint name"):
        landmarks.get_joint("Nope")
    with pytest.raises(ValueError, match="Unknown finger name"):
        landmarks.get_finger("palm")


def test_hand_frame_joint_and_finger_passthrough() -> None:
    frame = _sample_frame()

    assert frame.get_joint(JointName.MIDDLE_TIP) == (12.0, 12.1, 12.2)
    ring = frame.get_finger("ring")
    assert ring[JointName.RING_TIP] == (16.0, 16.1, 16.2)
