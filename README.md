# Hand Tracking SDK

Python SDK for consuming telemetry from [Hand Tracking Streamer (HTS)](https://github.com/wengmister/hand-tracking-streamer).

## Overview

HTS streams UTF-8 CSV lines for wrist pose and hand landmarks.  
This SDK provides typed parsing and validation for:
- wrist packets: 7 floats (`x, y, z, qx, qy, qz, qw`)
- landmark packets: 63 floats (`21 x [x, y, z]`)

Streamed joints are in Mediapipe-style 21 landmark points.

>[!IMPORTANT]
> Pre-release: This library is under active development. Expect breaking changes to come.

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

## Optional Visualization (Rerun)

Install visualization extras:

```bash
pip install "hand-tracking-sdk[visualization]"
```

Basic usage:

```python
from hand_tracking_sdk import HTSClient, HTSClientConfig, RerunVisualizer, StreamOutput

client = HTSClient(HTSClientConfig(output=StreamOutput.BOTH))
visualizer = RerunVisualizer()

for event in client.iter_events():
    visualizer.log_event(event)
```

CLI example script:

```bash
uv run --with rerun-sdk python examples/visualize_rerun.py --transport tcp_server --host 0.0.0.0 --port 8000
```

By default, the visualizer converts Unity left-handed coordinates to right-handed view.
Landmarks are treated as wrist-relative and rendered in the corresponding wrist frame.
Landmark colors are per hand by default: left=blue, right=red.
The 3D view uses a dark background and FLU frame (`x=forward, y=left, z=up`).

Other example scripts:
- frame-only stream summary:
  `uv run python examples/stream_frames.py --transport tcp_server --host 0.0.0.0 --port 8000`
- JSONL capture for replay/analysis:
  `uv run python examples/log_to_jsonl.py --transport tcp_server --host 0.0.0.0 --port 8000 --output both --path runs/hand_tracking.jsonl`

## Streaming Client API

`HTSClient` provides a high-level sync stream with filtering and error policy controls.

Key controls:
- transport mode: `udp`, `tcp_server`, `tcp_client`
- output mode: `packets`, `frames`, `both`
- hand filter: `left`, `right`, `both`
- parse error policy: `strict` (raise) or `tolerant` (skip malformed lines)

```python
from hand_tracking_sdk import HTSClient, HTSClientConfig, StreamOutput

client = HTSClient(HTSClientConfig(output=StreamOutput.BOTH))
for event in client.iter_events():
    print(event)
```

Note: `StreamOutput.FRAMES` emits only when both wrist *and* landmarks packets have been received for a hand side.
Note: default `error_policy` is `strict`; use `tolerant` to skip malformed lines instead of stopping the stream.

## Observability and Error Model

`HTSClient` exposes structured counters and optional log hooks:
- `get_stats()` / `reset_stats()`
- `ClientStats` counters:
  - `lines_received`, `parse_errors`, `dropped_lines`
  - `packets_filtered`, `packets_emitted`, `frames_emitted`
  - `callbacks_invoked`, `callback_errors`
- `log_hook` in `HTSClientConfig` receives `StreamLogEvent` values (`LogEventKind`)

Additional high-level client exceptions:
- `ClientConfigurationError`
- `ClientCallbackError`

Core SDK types also include these fields and helpers for further integration:
- `HandFrame.frame_id` (explicit frame identifier)
- timestamps for receive/source tracking:
  - `recv_ts_ns`
  - `recv_time_unix_ns`
  - `source_ts_ns`
- deterministic serialization helpers:
  - `WristPose.to_dict()` / `WristPose.from_dict()`
  - `HandLandmarks.to_dict()` / `HandLandmarks.from_dict()`
  - `HandFrame.to_dict()` / `HandFrame.from_dict()`

`HandFrameAssembler` defaults to:
- left frame id: `hts_left_hand`
- right frame id: `hts_right_hand`

You can override these via `frame_id_by_side` for middleware-specific naming.

## Example

```python
from hand_tracking_sdk import parse_line

packet = parse_line("Right wrist:, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0")
print(packet.side, packet.kind, packet.data)
```

The parser validates packet label, side, and value count, and raises `ParseError` on malformed input.

## Protocol Reference

- [`hand-tracking-streamer/README.md`](https://github.com/wengmister/hand-tracking-streamer/blob/main/README.md)
- [`hand-tracking-streamer/CONNECTIONS.md`](https://github.com/wengmister/hand-tracking-streamer/blob/main/CONNECTIONS.md)

## Documentation

- Sphinx source lives in `docs/`
- Read the Docs build config: `.readthedocs.yaml`
- Local build command:

```bash
sphinx-build -b html docs docs/_build/html
```

## License
Apache-2.0
