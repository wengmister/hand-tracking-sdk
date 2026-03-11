<div align="center">
  <img width="512" height="362" alt="sdk_logo" src="https://github.com/user-attachments/assets/33cbe2c6-2da9-4868-b5f0-c66e7abc6e3e" />
  <h3 align="center">
    Python SDK for consuming telemetry from
    <a href="https://github.com/wengmister/hand-tracking-streamer">Hand Tracking Streamer (HTS)</a>
  </h3>
</div>
<p align="center">

  <a href="https://www.meta.com/experiences/hand-tracking-streamer/26303946202523164/">
   <img src="https://img.shields.io/badge/VR_app-Meta_Quest_Store-FF5757?labelColor=grey" alt="Horizon Store Release">
  </a>

  <a href="https://github.com/wengmister/hand-tracking-streamer">
    <img src="https://img.shields.io/badge/VR_app-GitHub-orange?labelColor=grey" alt="Python SDK">
  </a>

  <a href="https://github.com/wengmister/hand-tracking-streamer/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%20License%202.0-yellow.svg" alt="Apache 2.0">
  </a>

  <a href="https://hand-tracking-sdk.readthedocs.io/">
    <img src="https://img.shields.io/badge/API-ReadTheDocs-green.svg" alt="API Documentation">
  </a>


</p>

**Hand Tracking SDK** is a Python package for consuming HTS hand-tracking telemetry (UDP/TCP), parsing wrist/landmark data into typed frames, and providing conversion, visualization, and integration-ready APIs.

This SDK is hosted on [PyPI](https://pypi.org/project/hand-tracking-sdk/), with API documentation [Here](https://hand-tracking-sdk.readthedocs.io/)

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

## Streaming Client

`HTSClient` provides a high-level sync stream with filtering and error policy controls.

- **Transport**: UDP, TCP server, TCP client
- **Output**: raw packets, assembled frames, or both
- **Hand filter**: left, right, or both
- **Error policy**: strict (raise) or tolerant (skip malformed)
- **Observability**: `get_stats()` counters, `log_hook` for structured events

### Frame Assembly

`HandFrameAssembler` correlates wrist + landmark packets into per-hand `HandFrame`
objects. Head pose packets produce `HeadFrame` events. Stale out-of-order updates
are discarded.

A `HandFrame` includes:
- `side`: Left or Right
- `wrist`: `WristPose(x, y, z, qx, qy, qz, qw)`
- `landmarks.points`: 21 MediaPipe-style `(x, y, z)` joints
- Per-joint access: `frame.get_joint(JointName.INDEX_TIP)`
- Per-finger access: `frame.get_finger("index")`
- Timing metadata: `recv_ts_ns`, `source_ts_ns`, `sequence_id`

### Coordinate Conversion

Explicit Unity left-handed to right-handed converters:

```python
from hand_tracking_sdk.convert import (
    convert_hand_frame_unity_left_to_right,
    unity_left_to_rfu_position,          # right-forward-up
    unity_left_to_rfu_rotation_matrix,
)
```

### Joint & Finger Access

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

### Telemetry

| Script | Description |
|--------|-------------|
| `examples/visualize_rerun.py` | Rerun 3D visualization with coordinate frames and jitter metrics |
| `examples/stream_frames.py` | Print assembled frames to console |
| `examples/log_to_jsonl.py` | JSONL capture for replay and analysis |
| `examples/jitter_report.py` | Timing jitter report |

```bash
uv run python examples/visualize_rerun.py --transport tcp_server --host 0.0.0.0 --port 8000
```

### Video Host (WebRTC)

Host-side scripts that stream video back to the Quest headset over WebRTC.
See [`examples/video/README.md`](examples/video/README.md) for details.

| Script | Source | Description |
|--------|--------|-------------|
| `test_pattern_video_host.py` | Test pattern | Colour bars — no hardware needed |
| `webcam_video_host.py` | USB webcam | Streams a local camera feed |
| `inspire_hand_video_host.py` | MuJoCo | Bimanual Inspire Hand with vector retargeting |
| `shadow_hand_video_host.py` | MuJoCo | Bimanual Shadow Hand E3M5 with vector retargeting |
| `aloha_video_host.py` | MuJoCo | ALOHA 2 bimanual arms with IK |

```bash
# Test pattern — no extra dependencies:
uv run examples/video/test_pattern_video_host.py

# Shadow Hand bimanual retargeting:
uv run examples/video/shadow_hand_video_host.py --mocap-tcp-port 5555
```

### Simulation Teleop

The MuJoCo video hosts close a full teleoperation loop: Quest sends hand + head
mocap over TCP, Python drives a MuJoCo simulation, and the rendered camera view
streams back to the headset over WebRTC.

```
Quest 3/3S ──TCP──► HTSClient ──► MuJoCo pre_step + mj_step + render
                                                     │
Quest 3/3S ◄────────────── WebRTC H.264 ◄────────────┘
```

## Protocol and Docs

- HTS protocol: [hand-tracking-streamer README](https://github.com/wengmister/hand-tracking-streamer/blob/main/README.md)
- SDK API docs: [hand-tracking-sdk.readthedocs.io](https://hand-tracking-sdk.readthedocs.io/)

## License

Apache-2.0
