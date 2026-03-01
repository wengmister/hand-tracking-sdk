# Video Streaming Examples

Host-side examples for streaming video over WebRTC to a Quest client.

## Sources

| Script | Source | Description |
|--------|--------|-------------|
| `mujoco_video_host.py` | MuJoCo sim | ALOHA bimanual robot with optional mocap-driven teleop via mink IK |
| `webcam_video_host.py` | Webcam | Streams a local webcam feed |
| `test_pattern_video_host.py` | Synthetic | Color-bar test pattern for connectivity debugging |

## Presets

All hosts accept a `--preset` flag to configure resolution (FPS is best-effort):

| Preset | Resolution |
|--------|-----------|
| `480p` | 640 x 480 |
| `720p` | 1280 x 720 |
| `1080p` | 1920 x 1080 |

## MuJoCo Offscreen Framebuffer

MuJoCo's offscreen renderer has a default framebuffer size of **640 x 480**.
If the requested preset resolution exceeds this, the renderer will raise an
error like:

```
Image width 1280 > framebuffer width 640
```

To fix this, set `offwidth` and `offheight` in the model XML's `<visual>`
section to at least the maximum resolution you intend to use:

```xml
<visual>
  <global offwidth="1920" offheight="1080"/>
</visual>
```

The bundled ALOHA scene (`assets/aloha/scene.xml`) already sets this to
1920 x 1080, covering all presets up to 1080p.

## MuJoCo Physics Timing

The MuJoCo source adapter synchronizes physics to wall-clock time.  Each
rendered frame advances the simulation by
`round(elapsed_wall_time / model.opt.timestep)` physics steps, capped at 2x
the target frame interval to prevent runaway catch-up after render hiccups.

Frame pacing uses `time.sleep` in the render worker thread, which has sub-ms
resolution on Python 3.12+ Windows (via high-resolution waitable timers).
This avoids the ~15.6 ms granularity of `asyncio.sleep` on Windows.
