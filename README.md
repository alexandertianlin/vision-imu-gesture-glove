# Vision-Aided IMU Gesture Glove

This repository contains a real-time gesture-glove prototype that combines Unity, wearable IMU/tactile sensing, and MediaPipe Hand visual correction.

The full project archive is included in `release/vision-imu-gesture-glove.zip`. Extract it to get:

- Unity source project under `unity/`
- Python MediaPipe UDP sender under `python/`
- project screenshot and documentation under `docs/`

![System demo](docs/images/system-demo.png)

## Highlights

- Unity virtual hand driven by serial IMU and tactile sensor data.
- Python MediaPipe Hand pipeline for per-finger open/fist recognition.
- UDP bridge from visual detection to Unity correction.
- Palm-facing open-hand reset for IMU drift mitigation.
- Per-finger visual hold watchdog to avoid short IDLE spikes cancelling correction.
- Diagnostic logging for UDP receive, JSON parse, filtering, takeover, application, and cancellation.

## Archive

Download `release/vision-imu-gesture-glove.zip` and extract it locally. The archive excludes Unity generated folders such as `Library`, `Logs`, `.vs`, and Python `.venv`.

## Demo

The screenshot below shows the phone camera recognition window on the left and Unity virtual hand visualization on the right.

![System demo](docs/images/system-demo.png)
