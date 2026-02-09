from hand_tracking_sdk import HandFrameAssembler, HandSide


def test_frame_emits_only_when_both_components_present() -> None:
    assembler = HandFrameAssembler()

    no_frame = assembler.push_line(
        "Right wrist:, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0",
        recv_ts_ns=100,
        recv_time_unix_ns=1_000,
    )
    assert no_frame is None

    frame = assembler.push_line(
        "Right landmarks:, " + ", ".join(str(i / 100.0) for i in range(63)),
        recv_ts_ns=120,
        recv_time_unix_ns=1_020,
    )

    assert frame is not None
    assert frame.side == HandSide.RIGHT
    assert frame.sequence_id == 0
    assert frame.recv_ts_ns == 120
    assert frame.wrist_recv_ts_ns == 100
    assert frame.landmarks_recv_ts_ns == 120


def test_sequence_increments_on_new_component_update() -> None:
    assembler = HandFrameAssembler()

    assembler.push_line("Left wrist:, 1, 2, 3, 4, 5, 6, 7", recv_ts_ns=10, recv_time_unix_ns=10)
    first = assembler.push_line(
        "Left landmarks:, " + ", ".join(str(i) for i in range(63)),
        recv_ts_ns=20,
        recv_time_unix_ns=20,
    )

    assert first is not None
    assert first.sequence_id == 0

    second = assembler.push_line(
        "Left wrist:, 7, 6, 5, 4, 3, 2, 1",
        recv_ts_ns=30,
        recv_time_unix_ns=30,
    )

    assert second is not None
    assert second.sequence_id == 1
    assert second.recv_ts_ns == 30


def test_stale_updates_are_ignored() -> None:
    assembler = HandFrameAssembler()

    assembler.push_line("Right wrist:, 1, 2, 3, 4, 5, 6, 7", recv_ts_ns=100, recv_time_unix_ns=100)
    first = assembler.push_line(
        "Right landmarks:, " + ", ".join(str(i) for i in range(63)),
        recv_ts_ns=120,
        recv_time_unix_ns=120,
    )
    assert first is not None

    stale = assembler.push_line(
        "Right wrist:, 9, 9, 9, 9, 9, 9, 9",
        recv_ts_ns=90,
        recv_time_unix_ns=90,
    )
    assert stale is None

    fresh = assembler.push_line(
        "Right wrist:, 9, 9, 9, 9, 9, 9, 9",
        recv_ts_ns=130,
        recv_time_unix_ns=130,
    )
    assert fresh is not None
    assert fresh.sequence_id == 1
    assert fresh.wrist_recv_ts_ns == 130


def test_sequence_is_independent_per_hand_side() -> None:
    assembler = HandFrameAssembler()
    landmarks_line = "Left landmarks:, " + ", ".join(str(i) for i in range(63))
    right_landmarks_line = "Right landmarks:, " + ", ".join(str(i) for i in range(63))

    assembler.push_line("Left wrist:, 1, 2, 3, 4, 5, 6, 7", recv_ts_ns=10, recv_time_unix_ns=10)
    left_frame = assembler.push_line(landmarks_line, recv_ts_ns=20, recv_time_unix_ns=20)

    assembler.push_line("Right wrist:, 1, 2, 3, 4, 5, 6, 7", recv_ts_ns=11, recv_time_unix_ns=11)
    right_frame = assembler.push_line(right_landmarks_line, recv_ts_ns=21, recv_time_unix_ns=21)

    assert left_frame is not None
    assert right_frame is not None
    assert left_frame.sequence_id == 0
    assert right_frame.sequence_id == 0


def test_default_timestamps_are_generated() -> None:
    assembler = HandFrameAssembler(include_wall_time=True)

    assembler.push_line("Left wrist:, 1, 2, 3, 4, 5, 6, 7")
    frame = assembler.push_line("Left landmarks:, " + ", ".join(str(i) for i in range(63)))

    assert frame is not None
    assert frame.recv_ts_ns > 0
    assert frame.recv_time_unix_ns is not None
    assert frame.recv_time_unix_ns > 0
