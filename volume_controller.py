import cv2
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import mediapipe as mp

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]
max_vol = volume_range[1]

volBar = 400

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if hasattr(mp_hands.HandLandmark, 'THUMB_TIP'):
                thumb_x, thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                index_x, index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                length = math.hypot(thumb_x - index_x, thumb_y - index_y)

                vol = np.interp(length, [0, 0.3], [min_vol, max_vol])
                volBar = np.interp(length, [0, 0.3], [400, 150])
                volPer = np.interp(length, [0, 0.3], [0, 100])

                volume.SetMasterVolumeLevel(vol, None)

                cv2.circle(img, (int(thumb_x * wCam), int(thumb_y * hCam)), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (int(index_x * wCam), int(index_y * hCam)), 10, (0, 255, 0), cv2.FILLED)
                cv2.line(img, (int(thumb_x * wCam), int(thumb_y * hCam)), (int(index_x * wCam), int(index_y * hCam)), (0, 255, 0), 3)
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
