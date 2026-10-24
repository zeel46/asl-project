import json
import os
import shutil
import zipfile

import cv2
import numpy as np

from word_features import create_holistic, frame_points, sequence_features
from words_config import WORDS, REPLIES

ZIP_PATH = "archive.zip"
VIDEO_DIR = "wlasl_words"
CLIP_DIR = "static/clips"
OUTPUT = "word_sequences.npz"

archive = zipfile.ZipFile(ZIP_PATH)
names = set(archive.namelist())
glosses = {e["gloss"]: e["instances"] for e in json.load(archive.open("WLASL_v0.3.json"))}
needed = sorted(set(WORDS) | {r for rs in REPLIES.values() for r in rs})

videos = {}
for word in needed:
    folder = os.path.join(VIDEO_DIR, word)
    os.makedirs(folder, exist_ok=True)
    videos[word] = []
    for inst in glosses[word]:
        member = f"videos/{inst['video_id']}.mp4"
        if member not in names:
            continue
        path = os.path.join(folder, f"{inst['video_id']}.mp4")
        if not os.path.exists(path):
            with archive.open(member) as src, open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        videos[word].append((path, inst["signer_id"]))
    print(f"{word}: {len(videos[word])} videos", flush=True)

def video_width(path):
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    return width


os.makedirs(CLIP_DIR, exist_ok=True)
for word in needed:
    clip = os.path.join(CLIP_DIR, word.replace(" ", "_") + ".mp4")
    sharpest = max((p for p, _ in videos[word]), key=video_width)
    shutil.copy(sharpest, clip)

holistic = create_holistic()
X, y, signers, files = [], [], [], []
for word in WORDS:
    for path, signer in videos[word]:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            frames.append(frame_points(frame, holistic))
        cap.release()
        seq = sequence_features(frames)
        if seq is None:
            print(f"  skipped {path} (no hands found)", flush=True)
            continue
        X.append(seq)
        y.append(word)
        signers.append(signer)
        files.append(path)
    print(f"features done: {word}", flush=True)

np.savez(OUTPUT, X=np.array(X), y=np.array(y), signers=np.array(signers), files=np.array(files))
print(f"Saved {OUTPUT}: {len(y)} sequences")
