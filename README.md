# Hand Tracking SDK

Python SDK for consuming telemetry from [Hand Tracking Streamer (HTS)](https://github.com/wengmister/hand-tracking-streamer).

## Overview

HTS streams UTF-8 CSV lines for wrist pose and hand landmarks.  
This SDK provides typed parsing and validation for:
- wrist packets: 7 floats (`x, y, z, qx, qy, qz, qw`)
- landmark packets: 63 floats (`21 x [x, y, z]`)
- optional head pose packets: 7 floats (`x, y, z, qx, qy, qz, qw`)

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
- Head pose packets are parsed but intentionally excluded from hand frame assembly.
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

The rerun example now also logs jitter/drop metrics under `metrics/jitter/<side>/...`
when `--show-jitter-panel` is enabled.

By default, the visualizer converts Unity left-handed coordinates to right-handed view.
Landmarks are treated as wrist-relative and rendered in the corresponding wrist frame.
Landmark colors are per hand by default: left=blue, right=red.
The 3D view uses a dark background and FLU frame (`x=forward, y=left, z=up`).

Other example scripts:
- frame-only stream summary:
  `uv run python examples/stream_frames.py --transport tcp_server --host 0.0.0.0 --port 8000`
- JSONL capture for replay/analysis:
  `uv run python examples/log_to_jsonl.py --transport tcp_server --host 0.0.0.0 --port 8000 --output both --path runs/hand_tracking.jsonl`

## Optional Video Host Service (WebRTC + WebSocket)

Install video extras:

```bash
pip install "hand-tracking-sdk[video]"
```

Run host signaling/media service:

```bash
uv run --with "hand-tracking-sdk[video]" python examples/video_host_service.py --signaling-host 0.0.0.0 --signaling-port 8765 --source test --preset 720p30
```

For handshake/signaling diagnostics:

```bash
uv run --with "hand-tracking-sdk[video]" python examples/video_host_service.py --signaling-host 0.0.0.0 --signaling-port 8765 --source test --preset 720p30 --verbose
```

This service exposes:
- control/signaling over WebSocket (`type`, `session_id`, `payload`)
- one outbound WebRTC H.264 video stream
- test pattern or webcam source adapters

## Streaming Client API

`HTSClient` provides a high-level sync stream with filtering and error policy controls.

Key controls:
- transport mode: `udp`, `tcp_server`, `tcp_client`
- output mode: `packets`, `frames`, `frames_all`, `both`
- hand filter: `left`, `right`, `both`
- parse error policy: `strict` (raise) or `tolerant` (skip malformed lines)

```python
from hand_tracking_sdk import HTSClient, HTSClientConfig, StreamOutput

client = HTSClient(
    HTSClientConfig(
        output=StreamOutput.BOTH,  # packets + assembled frames
    )
)

for event in client.iter_events():
    print(event)
```

## What HTS Sends

HTS emits UTF-8 CSV lines:
- wrist packet: 7 floats (`x, y, z, qx, qy, qz, qw`)
- landmarks packet: 63 floats (`21 x [x, y, z]`)
- optional head pose packet: 7 floats (`x, y, z, qx, qy, qz, qw`)

Head pose data is optional and appears only when the sender enables head tracking.
Hand-only streams (wrist + landmarks) remain fully supported.

The SDK validates packet labels, hand side, and exact value counts.

## Core APIs

- Streaming client:
  - `HTSClient`, `HTSClientConfig`
  - output modes: `packets`, `frames`, `both`
  - transport modes: `udp`, `tcp_server`, `tcp_client`
- Frame assembly:
  - `HandFrameAssembler`
  - emits a frame only when both wrist and landmarks are available
  - optional `HeadFrame` emission in `frames_all` mode
- Coordinate conversion:
  - `unity_left_to_right_position`
  - `unity_left_to_right_quaternion`
  - `convert_wrist_pose_unity_left_to_right`
  - `convert_landmarks_unity_left_to_right`
  - `convert_hand_frame_unity_left_to_right`
- Parsing:
  - `parse_line`

## Observability

`HTSClient` exposes runtime counters and structured log hooks:
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
  - `WristPose.to_dict()` / `from_dict()`
  - `HandLandmarks.to_dict()` / `from_dict()`
  - `HandFrame.to_dict()` / `from_dict()`

## Frame Structure

A `HandFrame` includes:
- `side`: `Left` or `Right`
- `frame_id`: frame name for downstream systems
- `wrist`: `WristPose(x, y, z, qx, qy, qz, qw)`
- `landmarks.points`: tuple of 21 `(x, y, z)` points
- timing/sequence metadata (`sequence_id`, `recv_ts_ns`, `recv_time_unix_ns`, `source_ts_ns`)

To get telemetry for a specific joint from a frame, use `get_joint(...)`.
Joint names and order follow the HTS streamed contract (wrist is `JointName.WRIST`).

```python
from hand_tracking_sdk import HTSClient, HTSClientConfig, JointName, StreamOutput

client = HTSClient(HTSClientConfig(output=StreamOutput.FRAMES))

for frame in client.iter_events():
    x, y, z = frame.get_joint(JointName.INDEX_TIP)
    print(
        f"side={frame.side.value} joint={JointName.INDEX_TIP.value} "
        f"xyz=({x:.5f}, {y:.5f}, {z:.5f}) recv_ts_ns={frame.recv_ts_ns}"
    )
```

You can also query by finger group:

```python
index_points = frame.get_finger("index")
# returns dict[JointName, tuple[float, float, float]]
# keys include JointName.INDEX_PROXIMAL, JointName.INDEX_TIP, ...
```

`frames_all` mode may also emit `HeadFrame` events with consistent metadata:
- `side=Head`
- `frame_id`
- `head` pose (`x, y, z, qx, qy, qz, qw`)
- `sequence_id`, `recv_ts_ns`, `recv_time_unix_ns`, `source_ts_ns`, `source_frame_seq`

## Examples

- Rerun visualization:
  - install extra: `pip install "hand-tracking-sdk[visualization]"`
  - `uv run python examples/visualize_rerun.py --transport tcp_server --host 0.0.0.0 --port 8000`
- Frame-only stream:
  - `uv run python examples/stream_frames.py --transport tcp_server --host 0.0.0.0 --port 8000`
  - frame mode intentionally emits hand frames only (head pose packets are ignored by frame assembly)
- JSONL logging:
  - `uv run python examples/log_to_jsonl.py --transport tcp_server --host 0.0.0.0 --port 8000 --output both --path runs/hand_tracking.jsonl`
  - packets mode now includes `packet_type=head_pose` when present

## Protocol and Docs

- HTS protocol reference:
  - [`hand-tracking-streamer/README.md`](https://github.com/wengmister/hand-tracking-streamer/blob/main/README.md)
  - [`hand-tracking-streamer/CONNECTIONS.md`](https://github.com/wengmister/hand-tracking-streamer/blob/main/CONNECTIONS.md)
- SDK docs:
  - https://hand-tracking-sdk.readthedocs.io/

## License

Apache-2.0
