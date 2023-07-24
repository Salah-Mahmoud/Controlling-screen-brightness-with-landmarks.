import cv2
import mediapipe as mp
import math
import numpy as np
import screen_brightness_control as sbc

mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Check if frame was read correctly
        if not ret:
            break

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Dictionary to store coordinates
        finger_knuckles = {
            'Thumb': None,
            'Index': None
        }

        # Right hand
        if results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks.landmark
            finger_knuckles['Thumb'] = (int(hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP].x * image.shape[1]),
                                        int(hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP].y * image.shape[0]))
            finger_knuckles['Index'] = (
                int(hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]),
                int(hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]))

        # Draw circles at finger knuckles
        for knuckle, coordinate in finger_knuckles.items():
            if coordinate:
                cv2.circle(image, coordinate, 8, (0, 255, 0), thickness=-1)

        # Draw line between thumb and index finger
        if finger_knuckles['Thumb'] and finger_knuckles['Index']:
            cv2.line(image, finger_knuckles['Thumb'], finger_knuckles['Index'], (0, 255, 0), 3)

            # Calculate the distance between thumb and index finger manually
            length = math.hypot(finger_knuckles['Index'][0] - finger_knuckles['Thumb'][0],
                                finger_knuckles['Index'][1] - finger_knuckles['Thumb'][1])
            if length < 50:
                cv2.line(image, finger_knuckles['Thumb'], finger_knuckles['Index'], (0, 0, 255), 3)

            bri = np.interp(length, [20, 220], [0, 100])
            print(np.round(bri))
            sbc.set_brightness(bri)

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
