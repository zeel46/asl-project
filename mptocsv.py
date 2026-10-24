import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

dataset_dir = "asl_subset"
output_csv = "asl_landmarks.csv"
failed_log = "failed_images.txt"

failed_count = 0
success_count = 0

with open(output_csv, mode='w', newline='') as f, open(failed_log, mode='w') as flog:
    writer = csv.writer(f)
    header = [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')] + ['label']
    writer.writerow(header)

    for letter in sorted(os.listdir(dataset_dir)):
        letter_path = os.path.join(dataset_dir, letter)
        if not os.path.isdir(letter_path):
            continue

        for img_name in os.listdir(letter_path):
            img_path = os.path.join(letter_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                failed_count += 1
                flog.write(f"{img_path} - could not read image\n")
                continue
            image = cv2.imread(img_path)
            image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_CUBIC)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                row = []
                for lm in landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(letter)
                writer.writerow(row)
                success_count += 1
            else:
                failed_count += 1
                flog.write(f"{img_path} - no hand detected\n")

print(f"Done. Success: {success_count}, Failed: {failed_count}")