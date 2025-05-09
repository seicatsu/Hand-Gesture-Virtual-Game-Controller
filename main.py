import cv2
import mediapipe as mp
from functions import get_extended_fingers_by_angle
import vgamepad as vg
import time

# Initialize virtual Xbox 360 controller
gamepad = vg.VX360Gamepad()

# Setup MediaPipe
cap = cv2.VideoCapture(0)
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Movement tracking
prev_x, prev_y = None, None

while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    frm = cv2.resize(frm, (640, 480))
    res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Reset joystick each frame
    gamepad.left_joystick(x_value=0, y_value=0)
    gamepad.right_joystick(x_value=0, y_value=0)
    gamepad.update()

    if res.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
            drawing.draw_landmarks(frm, hand_landmarks, hands.HAND_CONNECTIONS)

            hand_label = handedness.classification[0].label
            finger_states = get_extended_fingers_by_angle(hand_landmarks)
            extended_count = sum(finger_states)

            if hand_label == 'Right':
                # Fist = attack (right trigger)
                if extended_count == 0:
                    gamepad.right_trigger(value=255)
                    gamepad.update()
                    cv2.putText(frm, "RH: Attacking", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    gamepad.right_trigger(value=0)
                    gamepad.update()

            elif hand_label == 'Left':
                # Index only = block (left trigger)
                if finger_states == [False, True, False, False, False]:
                    gamepad.left_trigger(value=255)
                    gamepad.update()
                    cv2.putText(frm, "LH: Blocking", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                else:
                    gamepad.left_trigger(value=0)
                    gamepad.update()

                # Move right stick based on index finger tip
                index_finger_tip = hand_landmarks.landmark[8]
                screen_width, screen_height = 640, 480  # match resized frame
                x = int(index_finger_tip.x * screen_width)
                y = int(index_finger_tip.y * screen_height)

                if prev_x is not None and prev_y is not None:
                    dx = x - prev_x
                    dy = y - prev_y

                    # Scale movement for right stick (camera control)
                    cam_x = max(min(int(dx * 5000), 32767), -32768)
                    cam_y = max(min(int(-dy * 5000), 32767), -32768)

                    gamepad.right_joystick(x_value=cam_x, y_value=cam_y)
                    gamepad.update()
                    cv2.putText(frm, "LH: Looking", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None
        cv2.putText(frm, "NO HANDS - IDLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
