import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

SEQ_LEN = 32
FINGERTIPS = [4, 8, 12, 16, 20]
FEATURES_PER_HAND = 2 + 2 * len(FINGERTIPS)
REST_POSITION = (0.0, 2.5)


def create_holistic(static=False):
    return mp_holistic.Holistic(static_image_mode=static, model_complexity=1,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)


def _points(landmarks, w, h):
    if landmarks is None:
        return None
    pts = np.array([[lm.x * w / h, lm.y] for lm in landmarks.landmark])
    return pts


def frame_points(frame, holistic):
    h, w = frame.shape[:2]
    res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pose = _points(res.pose_landmarks, w, h)
    if pose is None:
        return None
    return {
        "shoulders": pose[[11, 12]],
        "left": _points(res.left_hand_landmarks, w, h),
        "right": _points(res.right_hand_landmarks, w, h),
        "raw": res,
    }


def _hand_features(hand, origin, scale, mirror):
    if hand is None:
        return [REST_POSITION[0], REST_POSITION[1]] + [0.0] * (2 * len(FINGERTIPS))
    hand = hand.copy()
    if mirror:
        hand[:, 0] = -hand[:, 0]
    wrist = (hand[0] - origin) / scale
    rel = hand[FINGERTIPS] - hand[0]
    size = np.sqrt(np.sum((hand - hand[0]) ** 2, axis=1)).max()
    rel = rel / (size if size > 0 else 1.0)
    return list(wrist) + list(rel.reshape(-1))


def dominant_side(frames):
    movement = {}
    for side in ("left", "right"):
        wrists = [f[side][0] for f in frames if f[side] is not None]
        if len(wrists) < 2:
            movement[side] = 0.0
        else:
            movement[side] = float(np.sum(np.sqrt(np.sum(np.diff(wrists, axis=0) ** 2, axis=1))))
    return "left" if movement["left"] > movement["right"] else "right"


def sequence_features(frames):
    frames = [f for f in frames if f is not None and (f["left"] is not None or f["right"] is not None)]
    if len(frames) < 4:
        return None
    dom = dominant_side(frames)
    other = "right" if dom == "left" else "left"
    mirror = dom == "left"
    rows = []
    for f in frames:
        s = f["shoulders"].copy()
        if mirror:
            s[:, 0] = -s[:, 0]
        origin = s.mean(axis=0)
        scale = np.sqrt(np.sum((s[0] - s[1]) ** 2)) or 1.0
        rows.append(_hand_features(f[dom], origin, scale, mirror) +
                    _hand_features(f[other], origin, scale, mirror))
    return resample(np.array(rows), SEQ_LEN)


def resample(seq, length):
    old_t = np.linspace(0, 1, len(seq))
    new_t = np.linspace(0, 1, length)
    return np.stack([np.interp(new_t, old_t, seq[:, j]) for j in range(seq.shape[1])], axis=1)
