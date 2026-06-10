import json
import math
import os
import socket
import time
import urllib.request

import cv2
import mediapipe as mp


RTSP_URL = "http://192.168.2.139:4747/video"
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5055

MIN_HAND_CONFIDENCE = 0.70
VIS_CONFIDENCE_THRESHOLD = 0.75
CALIBRATION_REPETITIONS = 10
STAGE_SECONDS = 20.0
STABLE_WINDOW_MS = 0
SEND_COOLDOWN_MS = 80
IDLE_SEND_INTERVAL_MS = 120
PRINT_SEND_LOG = True
HAND_SEND_COOLDOWN_MS = 600
CALIBRATION_DB_PATH = os.path.join(os.path.dirname(__file__), "finger_calibration_db.json")
DATABASE_LIMIT = 100
CALIBRATION_CLUSTER_FRACTION = 0.30
MIN_DATABASE_SAMPLES = 6

# Per-finger relative score:
# 0.0 means close to that finger's learned fist distance.
# 1.0 means close to that finger's learned open distance.
OPEN_SCORE_THRESHOLD = 0.68
FIST_SCORE_THRESHOLD = 0.32
MIN_RANGE_SPAN = 0.06

FINGER_ORDER = ["Thumb", "Index", "Middle", "Ring", "Little"]
FULL_OPEN_FINGERS = ["Index", "Middle", "Ring", "Little"]
PALM_FACING_NORMAL_SIGN = -1.0
PALM_ORIENTATION_THRESHOLD = 0.35
FINGERS = {
    "Thumb": {"index": 0, "mcp": 2, "pip": 3, "tip": 4, "color": (255, 0, 0)},
    "Index": {"index": 1, "mcp": 5, "pip": 6, "dip": 7, "tip": 8, "color": (0, 255, 0)},
    "Middle": {"index": 2, "mcp": 9, "pip": 10, "dip": 11, "tip": 12, "color": (0, 255, 255)},
    "Ring": {"index": 3, "mcp": 13, "pip": 14, "dip": 15, "tip": 16, "color": (255, 0, 255)},
    "Little": {"index": 4, "mcp": 17, "pip": 18, "dip": 19, "tip": 20, "color": (255, 255, 0)},
}


def ensure_model():
    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    return model_path


def vec3(landmark):
    return (float(landmark.x), float(landmark.y), float(landmark.z))


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return math.sqrt(dot(a, a))


def dist(a, b):
    return norm(sub(a, b))


def get_hand_confidence(result, hand_index):
    if result.handedness and result.handedness[hand_index]:
        return float(result.handedness[hand_index][0].score)
    return 1.0


def palm_width(points):
    return max(1e-6, dist(points[5], points[17]))


def palm_normal_z(points):
    wrist = points[0]
    index_mcp = points[5]
    little_mcp = points[17]
    return cross(sub(index_mcp, wrist), sub(little_mcp, wrist))[2]


def palm_facing_score(points):
    wrist = points[0]
    index_mcp = points[5]
    little_mcp = points[17]
    normal = cross(sub(index_mcp, wrist), sub(little_mcp, wrist))
    normal_len = max(1e-6, norm(normal))
    return PALM_FACING_NORMAL_SIGN * normal[2] / normal_len


def hand_orientation(points):
    score = palm_facing_score(points)
    if score > PALM_ORIENTATION_THRESHOLD:
        return "PALM"
    if score < -PALM_ORIENTATION_THRESHOLD:
        return "BACK"
    return "SIDE"


def is_palm_facing_camera(points):
    return hand_orientation(points) == "PALM"


def finger_distance(points, finger_name):
    finger = FINGERS[finger_name]
    if "dip" in finger:
        chain_length = (
            dist(points[finger["mcp"]], points[finger["pip"]])
            + dist(points[finger["pip"]], points[finger["dip"]])
            + dist(points[finger["dip"]], points[finger["tip"]])
        )
    else:
        chain_length = (
            dist(points[finger["mcp"]], points[finger["pip"]])
            + dist(points[finger["pip"]], points[finger["tip"]])
        )

    chain_length = max(1e-6, chain_length)
    tip_mcp = dist(points[finger["tip"]], points[finger["mcp"]]) / chain_length
    tip_pip = dist(points[finger["tip"]], points[finger["pip"]]) / chain_length
    return 0.80 * tip_mcp + 0.20 * tip_pip


def new_ranges():
    return {
        name: {"min": 999.0, "max": -999.0}
        for name in FINGER_ORDER
    }


def new_calibration_samples():
    return {name: [] for name in FINGER_ORDER}


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def trim_samples(values):
    values = [float(v) for v in values if isinstance(v, (int, float))]
    if len(values) <= DATABASE_LIMIT:
        return values
    return values[-DATABASE_LIMIT:]


def load_calibration_db():
    empty = {name: {"fist": [], "open": []} for name in FINGER_ORDER}
    if not os.path.exists(CALIBRATION_DB_PATH):
        return empty
    try:
        with open(CALIBRATION_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return empty

    for name in FINGER_ORDER:
        item = data.get(name, {})
        empty[name]["fist"] = trim_samples(item.get("fist", []))
        empty[name]["open"] = trim_samples(item.get("open", []))
    return empty


def save_calibration_db(database):
    try:
        with open(CALIBRATION_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(database, f, indent=2)
    except OSError as e:
        print(f"Calibration database save failed: {e}")


def merge_calibration_samples(database, calibration_samples):
    for name in FINGER_ORDER:
        samples = sorted(calibration_samples.get(name, []))
        if len(samples) < MIN_DATABASE_SAMPLES:
            continue

        cluster_size = max(3, int(len(samples) * CALIBRATION_CLUSTER_FRACTION))
        fist_samples = samples[:cluster_size]
        open_samples = samples[-cluster_size:]

        database[name]["fist"] = trim_samples(database[name]["fist"] + fist_samples)
        database[name]["open"] = trim_samples(database[name]["open"] + open_samples)


def database_ready(database, finger_name):
    item = database.get(finger_name, {})
    return (
        len(item.get("fist", [])) >= MIN_DATABASE_SAMPLES
        and len(item.get("open", [])) >= MIN_DATABASE_SAMPLES
    )


def update_range(ranges, finger_name, value):
    ranges[finger_name]["min"] = min(ranges[finger_name]["min"], value)
    ranges[finger_name]["max"] = max(ranges[finger_name]["max"], value)


def range_ready(ranges, finger_name):
    item = ranges[finger_name]
    return item["max"] - item["min"] >= MIN_RANGE_SPAN


def all_ranges_ready(ranges):
    return all(range_ready(ranges, name) for name in FINGER_ORDER)


def score_finger(ranges, database, finger_name, value):
    if database_ready(database, finger_name):
        item = database[finger_name]
        fist_anchor = median(item["fist"])
        open_anchor = median(item["open"])
        if open_anchor is not None and fist_anchor is not None and abs(open_anchor - fist_anchor) >= MIN_RANGE_SPAN:
            low_anchor = min(fist_anchor, open_anchor)
            high_anchor = max(fist_anchor, open_anchor)
            score = (value - low_anchor) / (high_anchor - low_anchor)
            return max(0.0, min(1.0, score))

    item = ranges[finger_name]
    span = max(MIN_RANGE_SPAN, item["max"] - item["min"])
    score = (value - item["min"]) / span
    return max(0.0, min(1.0, score))


def classify_score(score):
    if score >= OPEN_SCORE_THRESHOLD:
        return "OPEN", score
    if score <= FIST_SCORE_THRESHOLD:
        return "FIST", 1.0 - score
    return "IDLE", max(score, 1.0 - score)


def command_for_state(state):
    if state == "OPEN":
        return "FINGER_OPEN"
    if state == "FIST":
        return "FINGER_FIST"
    return "IDLE"


def send_finger_packet(sock, finger_name, state, confidence, stable_ms, score, ranges, sequence_id, vis_conf):
    finger = FINGERS[finger_name]
    command = command_for_state(state)
    payload = {
        "timestampMs": int(time.monotonic() * 1000),
        "sequenceId": sequence_id,
        "command": command,
        "gestureState": command,
        "fingerName": finger_name,
        "fingerIndex": finger["index"],
        "confidence": float(confidence),
        "vis_conf": float(vis_conf),
        "stableMs": int(stable_ms),
        "score": float(score),
        "range": ranges[finger_name],
    }
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendto(data, (UNITY_IP, UNITY_PORT))
    if PRINT_SEND_LOG and state != "IDLE":
        print(f"PY_SEND seq={sequence_id} finger={finger_name} state={state} score={score:.2f} conf={confidence:.2f}")


def send_hand_packet(sock, command, confidence, scores, sequence_id, is_palm_facing, vis_conf):
    payload = {
        "timestampMs": int(time.monotonic() * 1000),
        "sequenceId": sequence_id,
        "command": command,
        "gestureState": command,
        "fingerIndex": -1,
        "fingerName": "All",
        "confidence": float(confidence),
        "vis_conf": float(vis_conf),
        "stableMs": 0,
        "score": float(sum(scores.values()) / max(1, len(scores))),
        "isPalmFacing": bool(is_palm_facing),
    }
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendto(data, (UNITY_IP, UNITY_PORT))
    if PRINT_SEND_LOG:
        print(f"PY_SEND seq={sequence_id} hand={command} avgScore={payload['score']:.2f} conf={confidence:.2f} palmFacing={is_palm_facing}")


def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    for landmark in landmarks:
        cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 4, (0, 255, 0), cv2.FILLED)


def draw_calibration_overlay(frame, active_finger, remaining, ranges_ready_flag, landmarks):
    color = FINGERS[active_finger]["color"]
    h, w, _ = frame.shape
    tip = landmarks[FINGERS[active_finger]["tip"]]
    cv2.circle(frame, (int(tip.x * w), int(tip.y * h)), 14, color, cv2.FILLED)
    cv2.putText(frame, f"CALIBRATE {active_finger}: open-close {CALIBRATION_REPETITIONS} times", (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
    status = "OK" if ranges_ready_flag else "move wider"
    cv2.putText(frame, f"{remaining:.1f}s left  range={status}", (24, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)


def draw_runtime_overlay(frame, finger_states, finger_scores, orientation):
    y = 36
    palm_color = (0, 255, 0)
    if orientation == "BACK":
        palm_color = (0, 165, 255)
    elif orientation == "SIDE":
        palm_color = (255, 255, 0)
    cv2.putText(frame, f"Facing: {orientation}", (24, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, palm_color, 2)
    y += 26
    for name in FINGER_ORDER:
        state = finger_states.get(name, "IDLE")
        score = finger_scores.get(name, 0.0)
        color = (180, 180, 180)
        if state == "OPEN":
            color = (0, 255, 0)
        elif state == "FIST":
            color = (0, 0, 255)
        cv2.putText(frame, f"{name}: {state} {score:.2f}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2)
        y += 26


def main():
    model_path = ensure_model()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ranges = new_ranges()
    calibration_samples = new_calibration_samples()
    calibration_db = load_calibration_db()
    calibration_started = time.monotonic()
    total_calibration_seconds = STAGE_SECONDS * len(FINGER_ORDER)
    calibrated = False

    last_state = {name: "IDLE" for name in FINGER_ORDER}
    state_since_ms = {name: None for name in FINGER_ORDER}
    last_sent_ms = {name: 0 for name in FINGER_ORDER}
    last_idle_sent_ms = {name: 0 for name in FINGER_ORDER}
    last_hand_sent_ms = 0
    last_hand_command = "IDLE"
    sequence_id = 0

    base_options = mp.tasks.BaseOptions
    hand_landmarker = mp.tasks.vision.HandLandmarker
    hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
    running_mode = mp.tasks.vision.RunningMode

    options = hand_landmarker_options(
        base_options=base_options(model_asset_path=model_path),
        running_mode=running_mode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=MIN_HAND_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_CONFIDENCE,
        min_tracking_confidence=0.65,
    )

    print(f"MediaPipe per-finger anchor sender -> Unity UDP {UNITY_IP}:{UNITY_PORT}")
    print(f"Calibration: each finger gets its own stage. Move only that finger open-close {CALIBRATION_REPETITIONS} times.")
    print(f"Visual confidence gate: Unity should reject correction packets below {VIS_CONFIDENCE_THRESHOLD:.2f}.")
    print(f"Calibration database: {CALIBRATION_DB_PATH}")

    with hand_landmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            now_ms = int(time.monotonic() * 1000)
            elapsed = time.monotonic() - calibration_started
            result = None

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect_for_video(mp_image, now_ms)

            finger_states = {}
            finger_scores = {}

            if result.hand_landmarks and result.hand_world_landmarks:
                hand_index = 0
                hand_conf = get_hand_confidence(result, hand_index)
                points = [vec3(x) for x in result.hand_world_landmarks[hand_index]]
                orientation = hand_orientation(points)
                palm_facing = orientation == "PALM"
                draw_landmarks(frame, result.hand_landmarks[hand_index])

                if hand_conf >= MIN_HAND_CONFIDENCE and not calibrated:
                    stage_idx = min(int(elapsed / STAGE_SECONDS), len(FINGER_ORDER) - 1)
                    active_finger = FINGER_ORDER[stage_idx]
                    value = finger_distance(points, active_finger)
                    update_range(ranges, active_finger, value)
                    calibration_samples[active_finger].append(value)
                    stage_remaining = STAGE_SECONDS - (elapsed % STAGE_SECONDS)
                    draw_calibration_overlay(
                        frame,
                        active_finger,
                        stage_remaining,
                        range_ready(ranges, active_finger),
                        result.hand_landmarks[hand_index])

                    if elapsed >= total_calibration_seconds:
                        if all_ranges_ready(ranges):
                            merge_calibration_samples(calibration_db, calibration_samples)
                            save_calibration_db(calibration_db)
                            calibrated = True
                            print("Per-finger calibration finished.")
                            for name in FINGER_ORDER:
                                item = ranges[name]
                                db_item = calibration_db[name]
                                print(
                                    f"  {name}: min={item['min']:.3f}, max={item['max']:.3f}, "
                                    f"span={item['max'] - item['min']:.3f}, "
                                    f"dbFist={len(db_item['fist'])}, dbOpen={len(db_item['open'])}")
                        else:
                            print("Calibration range was too small. Restarting per-finger calibration.")
                            ranges = new_ranges()
                            calibration_samples = new_calibration_samples()
                            calibration_started = time.monotonic()

                elif hand_conf >= MIN_HAND_CONFIDENCE and calibrated:
                    for name in FINGER_ORDER:
                        value = finger_distance(points, name)
                        score = score_finger(ranges, calibration_db, name, value)
                        state, state_conf = classify_score(score)
                        confidence = min(hand_conf, state_conf)
                        finger_states[name] = state
                        finger_scores[name] = score
                        command_state = state
                        if orientation == "SIDE":
                            command_state = "IDLE"
                        elif orientation == "BACK" and state != "OPEN":
                            command_state = "IDLE"

                        if state != last_state[name]:
                            last_state[name] = state
                            state_since_ms[name] = now_ms
                        stable_ms = 0 if state_since_ms[name] is None else now_ms - state_since_ms[name]

                        should_send = (
                            command_state in ("OPEN", "FIST")
                            and confidence >= MIN_HAND_CONFIDENCE
                            and stable_ms >= STABLE_WINDOW_MS
                            and now_ms - last_sent_ms[name] >= SEND_COOLDOWN_MS
                        )

                        if should_send:
                            sequence_id += 1
                            send_finger_packet(sock, name, command_state, confidence, stable_ms, score, ranges, sequence_id, hand_conf)
                            last_sent_ms[name] = now_ms
                            print(f"Sent {name} {command_state}: confidence={confidence:.2f}, vis_conf={hand_conf:.2f}, stable={stable_ms}ms, score={score:.2f}, facing={orientation}")
                        elif command_state == "IDLE" and now_ms - last_idle_sent_ms[name] >= IDLE_SEND_INTERVAL_MS:
                            sequence_id += 1
                            send_finger_packet(sock, name, "IDLE", 1.0, stable_ms, score, ranges, sequence_id, hand_conf)
                            last_idle_sent_ms[name] = now_ms

                    hand_command = "IDLE"
                    if orientation == "PALM":
                        if all(finger_states.get(name) == "OPEN" for name in FULL_OPEN_FINGERS):
                            hand_command = "TRIGGER_OPEN"
                        elif all(finger_states.get(name) == "FIST" for name in FINGER_ORDER):
                            hand_command = "TRIGGER_FIST"

                    if (
                        hand_command != "IDLE"
                        and (hand_command != last_hand_command or now_ms - last_hand_sent_ms >= HAND_SEND_COOLDOWN_MS)
                    ):
                        sequence_id += 1
                        send_hand_packet(sock, hand_command, hand_conf, finger_scores, sequence_id, palm_facing, hand_conf)
                        last_hand_command = hand_command
                        last_hand_sent_ms = now_ms
                    elif hand_command == "IDLE":
                        last_hand_command = "IDLE"

                    draw_runtime_overlay(frame, finger_states, finger_scores, orientation)

            elif not calibrated:
                cv2.putText(frame, "Waiting for hand landmarks...", (24, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 165, 255), 2)

            cv2.imshow("iPhone DroidCam + Per-Finger Anchor Sender", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()


if __name__ == "__main__":
    main()
