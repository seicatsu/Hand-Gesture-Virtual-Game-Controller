import cv2
import mediapipe as mp
from functions import get_extended_fingers_by_angle
import vgamepad as vg
from collections import deque
import time


# Initialize virtual Xbox 360 controller
gamepad = vg.VX360Gamepad()

dx_buffer = deque(maxlen=5)
dy_buffer = deque(maxlen=5)

# Setup MediaPipe
cap = cv2.VideoCapture(0)
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands

# Movement tracking
prev_x, prev_y = None, None
prev_time = time.time()  # For FPS calculation

with hands.Hands(max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.7) as hand_obj:

    while True:
        _, frm = cap.read()
        frm = cv2.flip(frm, 1)
        frm = cv2.resize(frm, (640, 480))
        res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                drawing.draw_landmarks(frm, hand_landmarks, hands.HAND_CONNECTIONS)

                hand_label = handedness.classification[0].label
                finger_states = get_extended_fingers_by_angle(hand_landmarks)
                extended_count = sum(finger_states)

                if hand_label == 'Right':
                    # Fist = attack
                    if extended_count == 0:
                        gamepad.right_trigger(value=255)
                        cv2.putText(frm, "RH: Attacking", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        gamepad.right_trigger(value=0)
                    gamepad.update()

                elif hand_label == 'Left':
                    # Block gesture = only index extended
                    if finger_states == [False, True, False, False, False]:
                        gamepad.left_trigger(value=255)
                        cv2.putText(frm, "LH: Blocking", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    else:
                        gamepad.left_trigger(value=0)
                    gamepad.update()

                    # Control camera with left index finger
                    index_tip = hand_landmarks.landmark[8]
                    screen_w, screen_h = 640, 480
                    x = int(index_tip.x * screen_w)
                    y = int(index_tip.y * screen_h)

                    if prev_x is not None and prev_y is not None:
                        dx = x - prev_x
                        dy = y - prev_y

                        # Deadzone
                        if abs(dx) < 2: dx = 0
                        if abs(dy) < 2: dy = 0

                        dx_buffer.append(dx)
                        dy_buffer.append(dy)

                        avg_dx = sum(dx_buffer) / len(dx_buffer)
                        avg_dy = sum(dy_buffer) / len(dy_buffer)

                        cam_x = max(min(int(avg_dx * 7000), 32767), -32768)
                        cam_y = max(min(int(-avg_dy * 4000), 32767), -32768)

                        gamepad.right_joystick(x_value=cam_x, y_value=cam_y)
                        gamepad.update()
                        cv2.putText(frm, "LH: Looking", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                    prev_x, prev_y = x, y
        else:
            prev_x, prev_y = None, None
            gamepad.left_trigger(value=0)
            gamepad.right_trigger(value=0)
            gamepad.right_joystick(x_value=0, y_value=0)
            gamepad.update()
            cv2.putText(frm, "NO HANDS - IDLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)


        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

cap.release()
cv2.destroyAllWindows()
