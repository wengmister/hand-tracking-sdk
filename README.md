# Hand Tracking SDK

Python SDK for consuming telemetry from [Hand Tracking Streamer (HTS)](https://github.com/wengmister/hand-tracking-streamer).

## Overview

HTS streams UTF-8 CSV lines for wrist pose, hand landmarks, and head pose from
Meta Quest headsets. This SDK provides typed parsing, frame assembly, coordinate
conversion, and high-level streaming — plus WebRTC video hosts for closing a
full simulation teleop loop.

>[!IMPORTANT]
> Pre-release: This library is under active development. Expect breaking changes.

## Installation

```bash
pip install hand-tracking-sdk                      # core SDK
pip install "hand-tracking-sdk[visualization]"     # + Rerun 3D viewer
pip install "hand-tracking-sdk[video]"             # + WebRTC video host
pip install "hand-tracking-sdk[video,sim]"         # + MuJoCo simulation
```

## Quick Start

```python
from hand_tracking_sdk import HTSClient, HTSClientConfig, JointName, StreamOutput

client = HTSClient(HTSClientConfig(output=StreamOutput.FRAMES))

for frame in client.iter_events():
    x, y, z = frame.get_joint(JointName.INDEX_TIP)
    print(f"{frame.side.value} index tip: ({x:.3f}, {y:.3f}, {z:.3f})")
```

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
