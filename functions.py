import math
import numpy as np

def calculate_angle(a, b, c):

    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

    dot_product = np.dot(ba, bc)
    magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)

    if magnitude == 0:
        return 0.0

    cos_angle = np.clip(dot_product / magnitude, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))

def is_thumb_extended(lm):
    tip = lm[4]  # Thumb tip
    base = lm[5]  # Index MCP

    dx = tip.x - base.x
    dy = tip.y - base.y
    distance = math.hypot(dx, dy)

    return distance > 0.1  # Threshold, tune based on test




def get_extended_fingers_by_angle(hand_landmarks, threshold=165):
    lm = hand_landmarks.landmark
    angles = []

    # Replace thumb angle with distance check
    thumb_extended = is_thumb_extended(lm)

    # Index
    angle_index = calculate_angle(lm[5], lm[6], lm[8])
    angles.append(angle_index)

    # Middle
    angle_middle = calculate_angle(lm[9], lm[10], lm[12])
    angles.append(angle_middle)

    # Ring
    angle_ring = calculate_angle(lm[13], lm[14], lm[16])
    angles.append(angle_ring)

    # Pinky
    angle_pinky = calculate_angle(lm[17], lm[18], lm[20])
    angles.append(angle_pinky)

    finger_states = [thumb_extended] + [angle > threshold for angle in angles]
    return finger_states