# Video Streaming Beta Test

Pre-release test for the Quest-to-PC video streaming pipeline.


![beta instruction](https://github.com/user-attachments/assets/f12a3644-eda6-493e-acb8-ddf2dc014c9f)


## What you need

- Meta Quest 3 or 3S
- PC on the same network (Wi-Fi or USB)
- Python 3.10+, `adb`

## Setup

### 1. Quest app

Download the `.apk` from the `feat/videostream` branch and sideload:

```bash
adb install hand-tracking-streamer.apk
```

APK: [https://github.com/wengmister/hand-tracking-streamer/tree/feat/videostream/hand_tracking_streamer.apk](https://github.com/wengmister/hand-tracking-streamer/blob/feat/videostream/hand_tracking_streamer.apk)

### 2. SDK

```bash
git clone -b feat/videostream https://github.com/wengmister/hand-tracking-sdk.git
cd hand-tracking-sdk
pip install -e ".[video,sim]"
```

> Make sure both the Quest app and SDK are from `feat/videostream`.

### 3. Quest app settings

1. Open Hand Tracking Streamer on your Quest.
2. Enter your PC's IP and a port (e.g. `5555`).
3. Toggle on **Head Pose** and **Video Feed**.
4. Select **TCP Wireless** (or Wired).
5. Don't start yet — launch the host script first.

### 4. Run a host script

```bash
# Test pattern (no MuJoCo needed):
uv run examples/video/test_pattern_video_host.py --verbose

# Inspire Hand retargeting:
uv run examples/video/inspire_hand_video_host.py --mocap-tcp-port 5555 --verbose

# ALOHA 2 IK (needs mink):
uv run examples/video/aloha_video_host.py --mocap-tcp-port 5555 --verbose

# Shadow Hand retargeting:
uv run examples/video/shadow_hand_video_host.py --mocap-tcp-port 5555 --verbose
```

### 5. Start streaming

Once the host prints `signaling endpoint (TCP/WebSocket): ws://<HOST_IP>:8765`,
tap **Start** in the Quest app. You should see the video feed in the headset.

## Troubleshooting

- **App won't connect** — check PC IP/port, make sure firewall allows TCP on 5555 and 8765
- **Black video** — add `--verbose`, check for errors; try `--preset 480p`
- **Hands don't move** — `--mocap-tcp-port` must match the port in the Quest app
- **Laggy** — try USB tethering for wired TCP or `--preset 480p`

## Feedback

Let me know if this works for you or if you hit any snags! File issues at
https://github.com/wengmister/hand-tracking-sdk/issues or just message me directly.
Console output with `--verbose` is super helpful for debugging.
