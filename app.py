import os
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from dtw_knn import DTWKNN
from hand_tracking import create_detector, get_hand
from knn_scratch import DEPTH_PAIR, KNNClassifier, normalize_landmarks
from motion_letters import detect_motion_letter, is_moving
from word_features import create_holistic, frame_points, sequence_features
from words_config import REPLIES, REPLY_TEXT

LIVE_K = 5

app = Flask(__name__)
model = KNNClassifier.load("knn_model.npz")
model.k = LIVE_K
all_labels = sorted(set(model.y_train))
letters = [l for l in all_labels if l.isalpha()]
numbers = sorted(l for l in all_labels if l.isdigit())
models = {"letters": model.subset(letters), "numbers": model.subset(numbers)}
pair_model = KNNClassifier.load("knn_depth_pair_model.npz")

words_model = DTWKNN.load("words_model.npz") if os.path.exists("words_model.npz") else None
words = sorted(set(words_model.y_train)) if words_model else []

detector = create_detector()
holistic = create_holistic()
lock = threading.Lock()
motion_history = deque(maxlen=60)


def read_frame():
    data = np.frombuffer(request.files["frame"].read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def as_list(points):
    return None if points is None else points.tolist()


def as_array(points):
    return None if points is None else np.array(points)


def draw_points(landmarks):
    return None if landmarks is None else [[lm.x, lm.y] for lm in landmarks.landmark]


def clip_url(word):
    return "/static/clips/" + word.replace(" ", "_") + ".mp4"


@app.route("/")
def index():
    return render_template("index.html", words=words)


@app.route("/predict", methods=["POST"])
def predict():
    mode = request.form.get("mode", "letters")
    frame = read_frame()
    if frame is None:
        return jsonify(hand=False)

    with lock:
        raw, hand, hand_name = get_hand(frame, detector)
        if raw is None:
            return jsonify(hand=False)
        now = time.time()
        motion_history.append((now, raw.reshape(21, 3)[:, :2]))
        moving = is_moving(motion_history, now)
        motion = detect_motion_letter(list(motion_history), now) if mode == "letters" else None
        if motion:
            motion_history.clear()

    neighbors = models[mode].nearest_labels(normalize_landmarks(raw)).tolist()
    if mode == "letters" and sum(n in DEPTH_PAIR for n in neighbors) > len(neighbors) / 2:
        neighbors = pair_model.nearest_labels(normalize_landmarks(raw, with_depth=True)).tolist()
    return jsonify(hand=True, side=hand_name, neighbors=neighbors,
                   points=draw_points(hand), moving=bool(moving), motion=motion)


@app.route("/word_frame", methods=["POST"])
def word_frame():
    frame = read_frame()
    with lock:
        f = None if frame is None else frame_points(frame, holistic)
    if f is None:
        return jsonify(body=False)
    return jsonify(body=True, shoulders=as_list(f["shoulders"]), left=as_list(f["left"]),
                   right=as_list(f["right"]),
                   draw=[p for p in (draw_points(f["raw"].left_hand_landmarks),
                                     draw_points(f["raw"].right_hand_landmarks)) if p])


@app.route("/classify_word", methods=["POST"])
def classify_word():
    if words_model is None:
        return jsonify(ok=False, error="Words model not trained yet - run train_words.py")
    frames = [{"shoulders": np.array(f["shoulders"]), "left": as_array(f["left"]),
               "right": as_array(f["right"])} for f in request.get_json()["frames"]]
    seq = sequence_features(frames)
    if seq is None:
        return jsonify(ok=False, error="No hands seen - keep both shoulders and hands in view")
    ranked, distance = words_model.rank(seq)
    top = ranked[0][0]
    replies = [{"word": r, "text": REPLY_TEXT.get(r, r), "clip": clip_url(r)}
               for r in REPLIES.get(top, [])]
    return jsonify(ok=True, ranked=ranked, distance=round(float(distance), 3),
                   replies=replies, clip=clip_url(top))


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
