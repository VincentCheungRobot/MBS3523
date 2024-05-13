import cv2
import mediapipe as mp
import numpy as np
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
def count_ok_gestures(landmarks, image_width, image_height):
   thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x * image_width,
                         landmarks[mp_hands.HandLandmark.THUMB_TIP].y * image_height])
   index_finger_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height])
   distance = np.linalg.norm(thumb_tip - index_finger_tip)
   if distance < 100:
       return True
   else:
       return False
cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
   counter = 0
   while cam.isOpened():
       ret, img = cam.read()
       if not ret:
           continue


       imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       results = hands.process(imgRGB)


       if results.multi_hand_landmarks:
           for hand_landmarks in results.multi_hand_landmarks:
               mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
               if count_ok_gestures(hand_landmarks.landmark, img.shape[1], img.shape[0]):
                   counter += 1
       cv2.putText(img, f'Count: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


       cv2.imshow('Hand Gesture Recognition', img)
       if cv2.waitKey(5) & 0xFF == 27:
           break


cam.release()
cv2.destroyAllWindows()