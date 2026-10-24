import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


def create_detector(static=False):
    if static:
        return mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                              min_detection_confidence=0.3)
    return mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6,
                          min_tracking_confidence=0.5)


def get_hand(frame, detector):
    h, w = frame.shape[:2]
    result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        return None, None, None
    hand = result.multi_hand_landmarks[0]
    side = result.multi_handedness[0].classification[0].label
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
    pts[:, 0] *= w / h
    pts[:, 2] *= w / h
    if side == "Right":
        pts[:, 0] = -pts[:, 0]
    hand_name = "left" if side == "Right" else "right"
    return pts.reshape(63), hand, hand_name
