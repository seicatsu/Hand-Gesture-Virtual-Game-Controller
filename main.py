import cv2
import mediapipe as mp
import pyautogui
from functions import get_extended_fingers_by_angle  # new angle-based function

cap = cv2.VideoCapture(0)

drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
holding_click = False

while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    frm = cv2.resize(frm, (640, 480))

    res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
            drawing.draw_landmarks(frm, hand_landmarks, hands.HAND_CONNECTIONS)

            hand_label = handedness.classification[0].label
            finger_states = get_extended_fingers_by_angle(hand_landmarks)  # [thumb, index, middle, ring, pinky]
            extended_count = sum(finger_states)

            if hand_label == 'Right':
                # Fist = Hold bow (mouse down)
                if extended_count == 0:
                    cv2.putText(frm, "RH: Holding Bow", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if not holding_click:
                        pyautogui.mouseDown(button='left')
                        holding_click = True
                else:
                    if holding_click:
                        pyautogui.mouseUp(button='left')
                        holding_click = False

            elif hand_label == 'Left':
                # Only index extended = Block
                if finger_states == [False, True, False, False, False]:
                    cv2.putText(frm, "LH: Blocking", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    pyautogui.click(button='right')  # simulate right-click

                # Move cursor using index finger (regardless of other fingers)
                index_finger_tip = hand_landmarks.landmark[8]
                screen_width, screen_height = pyautogui.size()
                x = int(index_finger_tip.x * screen_width)
                y = int(index_finger_tip.y * screen_height)
                pyautogui.moveTo(x, y)
                cv2.putText(frm, "LH: Moving Cursor", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    else:
        # No hands detected
        cv2.putText(frm, "NO HANDS - IDLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
