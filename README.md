# Hand Tracking SDK

Python SDK for consuming telemetry from [Hand Tracking Streamer (HTS)](https://github.com/wengmister/hand-tracking-streamer).

## Installation

```bash
pip install hand-tracking-sdk
```

Optional visualization support with Rerun:

```bash
pip install "hand-tracking-sdk[visualization]"
```

## Quickstart

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

The SDK validates packet labels, hand side, and exact value counts.

## Core APIs

- Streaming client:
  - `HTSClient`, `HTSClientConfig`
  - output modes: `packets`, `frames`, `both`
  - transport modes: `udp`, `tcp_server`, `tcp_client`
- Frame assembly:
  - `HandFrameAssembler`
  - emits a frame only when both wrist and landmarks are available
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
- `ClientStats`
- `log_hook` via `HTSClientConfig`

## ROS2-Friendly Data Model

The SDK is designed so ROS2 adapters can be layered cleanly without a hard ROS dependency:
- explicit timestamps (`recv_ts_ns`, `recv_time_unix_ns`, `source_ts_ns`)
- explicit `frame_id` and per-side sequencing
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

## Examples

- Rerun visualization:
  - install extra: `pip install "hand-tracking-sdk[visualization]"`
  - `uv run python examples/visualize_rerun.py --transport tcp_server --host 0.0.0.0 --port 8000`
- Frame-only stream:
  - `uv run python examples/stream_frames.py --transport tcp_server --host 0.0.0.0 --port 8000`
- JSONL logging:
  - `uv run python examples/log_to_jsonl.py --transport tcp_server --host 0.0.0.0 --port 8000 --output both --path runs/hand_tracking.jsonl`

## Protocol and Docs

- HTS protocol reference:
  - [`hand-tracking-streamer/README.md`](https://github.com/wengmister/hand-tracking-streamer/blob/main/README.md)
  - [`hand-tracking-streamer/CONNECTIONS.md`](https://github.com/wengmister/hand-tracking-streamer/blob/main/CONNECTIONS.md)
- SDK docs:
  - https://hand-tracking-sdk.readthedocs.io/

## License

Apache-2.0
