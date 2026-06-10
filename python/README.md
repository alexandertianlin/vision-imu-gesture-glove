# Python Vision Sender

`mediapipe_udp_sender.py` reads an iPhone/DroidCam video stream, detects a hand with MediaPipe, classifies per-finger open/fist states, and sends UDP correction packets to Unity.

Default settings:

- camera stream: `http://192.168.2.139:4747/video`
- Unity IP: `127.0.0.1`
- Unity UDP port: `5055`

Edit these constants at the top of `mediapipe_udp_sender.py` if your camera or Unity port changes.

The script creates `finger_calibration_db.json` locally after calibration. The example file in this repository is included only to show the expected data shape.
