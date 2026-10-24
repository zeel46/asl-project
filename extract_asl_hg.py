import csv
import os
from multiprocessing import Pool

import cv2

from hand_tracking import create_detector, get_hand

DATASET_DIR = "asl-hg/asl_dataset"
OUTPUT_CSV = "asl_hg_landmarks.csv"
SKIP = {"0", "J", "Z"}
WORKERS = 4
HEADER = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label", "person"]


def extract_sign(label):
    detector = create_detector(static=True)
    rows, missed = [], 0
    for name in sorted(os.listdir(os.path.join(DATASET_DIR, label))):
        image = cv2.imread(os.path.join(DATASET_DIR, label, name))
        raw = None if image is None else get_hand(image, detector)[0]
        if raw is None:
            missed += 1
            continue
        rows.append([round(v, 6) for v in raw] + [label, name.split("_")[0]])
    detector.close()
    return label, rows, missed


def load_finished():
    if not os.path.exists(OUTPUT_CSV):
        return [], set()
    with open(OUTPUT_CSV, newline="") as f:
        rows = list(csv.reader(f))[1:]
    counts = {}
    for r in rows:
        counts[r[63]] = counts.get(r[63], 0) + 1
    files = {label: len(os.listdir(os.path.join(DATASET_DIR, label))) for label in counts}
    finished = {label for label, n in counts.items() if n >= files[label] * 0.9}
    return [r for r in rows if r[63] in finished], finished


if __name__ == "__main__":
    rows, finished = load_finished()
    with open(OUTPUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows([HEADER] + rows)
    print(f"Already finished: {sorted(finished)}", flush=True)

    todo = [l for l in sorted(os.listdir(DATASET_DIR)) if l not in SKIP and l not in finished]
    with Pool(WORKERS) as pool, open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for label, sign_rows, missed in pool.imap_unordered(extract_sign, todo):
            writer.writerows(sign_rows)
            f.flush()
            print(f"{label}: {len(sign_rows)} hands found, {missed} missed", flush=True)

    print("Saved", OUTPUT_CSV)
