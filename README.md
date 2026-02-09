# Hand Tracking SDK

Python SDK for consuming telemetry from Hand Tracking Streamer (HTS).

## Package Layout

- `src/hand_tracking_sdk/` core SDK package
- `tests/` unit tests

## Current Scaffold

Implemented baseline components:
- typed packet models for wrist and landmarks
- HTS line parser for labels and CSV payloads
- strict value-count validation (`wrist=7`, `landmarks=63`)
- parsing tests for valid and malformed input

## Protocol Reference

- `hand-tracking-streamer/README.md`
- `hand-tracking-streamer/CONNECTIONS.md`
