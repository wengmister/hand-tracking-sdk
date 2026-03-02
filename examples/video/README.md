# Video Examples

Host-side scripts that receive Quest hand/head tracking data and stream video
back to the headset over WebRTC.

## Setup

```bash
pip install hand-tracking-sdk[video]       # WebRTC + signaling
pip install hand-tracking-sdk[video,sim]   # + MuJoCo sim hosts
```

## Scripts

| Script | Source | Description |
|--------|--------|-------------|
| `test_pattern_video_host.py` | Test pattern | Synthetic colour bars — no hardware needed |
| `webcam_video_host.py` | USB webcam | Streams a local camera feed |
| `inspire_hand_video_host.py` | MuJoCo | Bimanual Inspire Hand with vector retargeting |
| `shadow_hand_video_host.py` | MuJoCo | Bimanual Shadow Hand E3M5 with vector retargeting |
| `aloha_video_host.py` | MuJoCo | ALOHA 2 bimanual arms with IK (requires `mink`) |

## Quick start

```bash
# Simplest — no dependencies beyond the SDK:
uv run examples/video/test_pattern_video_host.py

# MuJoCo hand retargeting (Shadow Hand):
uv run examples/video/shadow_hand_video_host.py --mocap-tcp-port 5555

# Common flags:
#   --tcp-port 8765         WebSocket signaling port
#   --mocap-tcp-port 5555   Quest telemetry TCP port
#   --preset 720p           Video resolution (480p / 720p / 1080p)
#   --verbose               Enable detailed logging
#   --perf                  Log per-frame timing (MuJoCo hosts only)
```

Point the Quest app's signaling URL at `ws://<HOST_IP>:8765`.

## Internal modules

| File | Purpose |
|------|---------|
| `_common.py` | Shared argument parsing, mocap pump, MuJoCo host runner |
| `_tracking.py` | Relative head camera and wrist tracking from mocap frames |
| `_retarget.py` | Lightweight vector-based finger retargeting for MuJoCo |

## Assets

MuJoCo XML models live under `assets/`:

- `assets/shadow_hand/` — Shadow Hand E3M5 (left, right teleop, bimanual scene)
- `assets/aloha/` — ALOHA 2 bimanual arm scene
- `assets/inspire/` — Inspire Hand (left, right, bimanual scenes)

Shadow Hand and ALOHA models are borrowed from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
and follow their respective [license](https://github.com/google-deepmind/mujoco_menagerie/blob/main/LICENSE).
Inspire Hand assets are provided by Inspire Robots and slightly modified for simulation purposes.