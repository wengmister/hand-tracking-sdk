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

## Protocol Reference

- `../hand-tracking-streamer/README.md`
- `../hand-tracking-streamer/CONNECTIONS.md`
