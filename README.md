# Hand Tracking SDK

Python SDK for consuming telemetry from [Hand Tracking Streamer (HTS)](https://github.com/wengmister/hand-tracking-streamer).

## Overview

HTS streams UTF-8 CSV lines for wrist pose and hand landmarks.  
This SDK provides typed parsing and validation for:
- wrist packets: 7 floats (`x, y, z, qx, qy, qz, qw`)
- landmark packets: 63 floats (`21 x [x, y, z]`)

Streamed joints are in Mediapipe-style 21 landmark points.

## Example

```python
from hand_tracking_sdk import parse_line

packet = parse_line("Right wrist:, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0")
print(packet.side, packet.kind, packet.data)
```

The parser validates packet label, side, and value count, and raises `ParseError` on malformed input.

## Transport API

The SDK includes socket transport primitives for line-based ingestion:
- `UDPLineReceiver`
- `TCPServerLineReceiver`
- `TCPClientLineReceiver`

Network timeout and disconnect behavior is reported through typed exceptions:
- `TransportTimeoutError`
- `TransportDisconnectedError`
- `TransportClosedError`

## Frame Assembly API

Use `HandFrameAssembler` to combine wrist and landmark packets into coherent per-hand frames.

- Emits a frame only after both components are present for a hand.
- Ignores stale out-of-order updates (older receive timestamps).
- Increments `sequence_id` per hand side on each newly emitted frame.
- Supports:
  - `recv_ts_ns` (monotonic receive time)
  - `recv_time_unix_ns` (optional wall-clock receive time)
  - `source_ts_ns` (optional upstream timestamp; currently caller-provided)

## Coordinate Conversion API

The SDK provides explicit Unity-left-handed to right-handed conversion helpers:
- `unity_left_to_right_position`
- `unity_left_to_right_quaternion`
- `convert_wrist_pose_unity_left_to_right`
- `convert_landmarks_unity_left_to_right`
- `convert_hand_frame_unity_left_to_right`

Current conversion profile:
- position: flip Y sign (`x, y, z -> x, -y, z`)
- orientation: basis transform equivalent to Y-axis reflection

```python
from hand_tracking_sdk import (
    HandFrameAssembler,
    convert_hand_frame_unity_left_to_right,
)

assembler = HandFrameAssembler()
frame = assembler.push_line("Right wrist:, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0")
if frame is not None:
    converted = convert_hand_frame_unity_left_to_right(frame)
```

## Protocol Reference

- `../hand-tracking-streamer/README.md`
- `../hand-tracking-streamer/CONNECTIONS.md`
