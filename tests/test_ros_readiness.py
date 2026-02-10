from hand_tracking_sdk import HandFrame, HandLandmarks, HandSide, WristPose


def test_wrist_pose_roundtrip_dict() -> None:
    pose = WristPose(x=1.0, y=2.0, z=3.0, qx=0.1, qy=0.2, qz=0.3, qw=0.9)
    serialized = pose.to_dict()
    restored = WristPose.from_dict(serialized)

    assert restored == pose


def test_landmarks_roundtrip_dict() -> None:
    landmarks = HandLandmarks(points=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    serialized = landmarks.to_dict()
    restored = HandLandmarks.from_dict(serialized)

    assert restored == landmarks


def test_hand_frame_roundtrip_dict() -> None:
    frame = HandFrame(
        side=HandSide.RIGHT,
        frame_id="right_hand_link",
        wrist=WristPose(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        landmarks=HandLandmarks(points=((1.0, 2.0, 3.0),)),
        sequence_id=12,
        recv_ts_ns=1000,
        recv_time_unix_ns=2000,
        source_ts_ns=3000,
        wrist_recv_ts_ns=900,
        landmarks_recv_ts_ns=950,
    )

    serialized = frame.to_dict()
    restored = HandFrame.from_dict(serialized)

    assert serialized["frame_id"] == "right_hand_link"
    assert restored == frame
