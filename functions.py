import math

def calculate_angle(a, b, c):
    """
    Returns angle (in degrees) between 3 points (in 2D or 3D).
    b is the vertex.
    """
    ba = [a.x - b.x, a.y - b.y, a.z - b.z]
    bc = [c.x - b.x, c.y - b.y, c.z - b.z]

    # Dot product and magnitude
    dot_product = sum(i * j for i, j in zip(ba, bc))
    magnitude_ba = math.sqrt(sum(i * i for i in ba))
    magnitude_bc = math.sqrt(sum(i * i for i in bc))

    if magnitude_ba * magnitude_bc == 0:
        return 0

    # Clamp value for acos
    cos_angle = max(min(dot_product / (magnitude_ba * magnitude_bc), 1.0), -1.0)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def get_extended_fingers_by_angle(hand_landmarks):
    """
    Detects extended fingers using joint angles.
    Returns a list: [thumb, index, middle, ring, pinky] as True/False.
    """
    lm = hand_landmarks.landmark
    angles = []

    # Index: MCP–PIP–TIP
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

    # Thumb: MCP–IP–TIP (or CMC–MCP–TIP)
    angle_thumb = calculate_angle(lm[2], lm[3], lm[4])
    angles.insert(0, angle_thumb)

    # Thresholding
    finger_states = [angle > 160 for angle in angles]  # You can tune this threshold

    return finger_states  # [thumb, index, middle, ring, pinky]

