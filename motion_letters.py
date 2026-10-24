import numpy as np

WINDOW_SECONDS = 2.5
MAX_MISSES = 2
MIN_FRAMES = 8

FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}
SHAPES = {
    "J": {"up": ["pinky"], "down": ["index", "middle", "ring"], "tip": 20},
    "Z": {"up": ["index"], "down": ["middle", "ring", "pinky"], "tip": 8},
}


def finger_up(pts, finger):
    pip, tip = FINGERS[finger]
    return np.linalg.norm(pts[tip] - pts[0]) > 1.15 * np.linalg.norm(pts[pip] - pts[0])


def has_shape(pts, letter):
    shape = SHAPES[letter]
    return (all(finger_up(pts, f) for f in shape["up"]) and
            not any(finger_up(pts, f) for f in shape["down"]))


def hand_size(pts):
    return np.sqrt(np.sum((pts - pts[0]) ** 2, axis=1)).max()


def direction_changes(values, min_step):
    changes, last_sign, anchor = 0, 0, values[0]
    for v in values[1:]:
        step = v - anchor
        if abs(step) < min_step:
            continue
        sign = 1 if step > 0 else -1
        if last_sign and sign != last_sign:
            changes += 1
        last_sign, anchor = sign, v
    return changes


def is_z(path, size):
    xs, ys = path[:, 0] / size, path[:, 1] / size
    length = np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2))
    return length > 1.8 and direction_changes(xs, 0.25) >= 2 and ys[-1] - ys[0] > 0.35


def is_j(path, size):
    xs, ys = path[:, 0] / size, path[:, 1] / size
    length = np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2))
    third = max(1, len(xs) // 3)
    return length > 0.8 and ys.max() - ys[0] > 0.3 and abs(xs[-1] - xs[-third]) > 0.2


def is_moving(history, now, span=0.3, threshold=0.25):
    recent = [p for t, p in history if now - t <= span]
    if len(recent) < 2:
        return False
    shift = np.linalg.norm(recent[-1][0] - recent[0][0])
    return shift / hand_size(recent[-1]) > threshold


def shape_run(recent, letter):
    run, misses = [], 0
    for t, p in reversed(recent):
        if has_shape(p, letter):
            run.append(p)
        else:
            misses += 1
            if misses > MAX_MISSES:
                break
    return run[::-1]


def detect_motion_letter(history, now):
    recent = [(t, p) for t, p in history if now - t <= WINDOW_SECONDS]
    for letter, check in (("Z", is_z), ("J", is_j)):
        frames = shape_run(recent, letter)
        if len(frames) < MIN_FRAMES:
            continue
        path = np.array([p[SHAPES[letter]["tip"]] for p in frames])
        size = np.median([hand_size(p) for p in frames])
        if check(path, size):
            return letter
    return None
