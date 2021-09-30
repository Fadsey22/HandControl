import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from mediapipe.python.solutions import holistic


def mpsu():
    mp_d = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

def capture():
    cap = cv2.VideoCapture(0) # open video 0 (if only one webcam)
    while cap.isOpened(): # while camera is on
        ret, frame = cap.read() # read feed from webcam
        cv2.imshow('Webcam feed',frame) # sets up display for webcam
        if cv2.waitKey(10) & 0xFF == ord('q'): # if q is pressed it cancels
            print("BYE")
            break
    cap.release()
    cv2.destroyAllWindows()
def detect():
    mpsu()
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    ft = 0
    LeftArm = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()  # read feed from webcam
            pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            val = holistic.process(pic)
            mp_drawing.draw_landmarks(pic, val.face_landmarks, mp_holistic.FACE_CONNECTIONS) # draws everything face related


            # Right hand
           # mp_drawing.draw_landmarks(pic, val.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # rhand related

            # Left Hand
           #mp_drawing.draw_landmarks(pic, val.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # lhand related
            # Pose Detections
            mp_drawing.draw_landmarks(pic, val.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # should related
            landmarks = val.pose_landmarks.landmark
            try:
                if ft != 0:
                    #print(prevWrist, landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y)
                    if abs(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y - prevLWrist) >= 0.1: # check if there is a difference between wrist position of left hand
                        cv2.putText(pic, "JAB", (80, 375), cv2.FONT_HERSHEY_COMPLEX, 4, (200, 0, 0), 10)
                        #print("JAB")
                    if abs(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y - prevRWrist) >= 0.1:
                        cv2.putText(pic, "CROSS", (80, 375), cv2.FONT_HERSHEY_COMPLEX, 4, (200, 0, 0), 10)

                prevLWrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y
                prevRWrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y
                ft = 1
                # LeftArm.append(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x)
                # LeftArm.append(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y)
               # print(LeftArm[0],LeftArm[1])
            except Exception as e:
                print(f"Cant find: {e}")


            cv2.imshow('Webcam feed', pic)  # sets up display for webcam
            if cv2.waitKey(10) & 0xFF == ord('q'):  # if q is pressed it cancels
                print("BYE")
                break
        cap.release()
        cv2.destroyAllWindows()

def handdetect():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip on horizontal
            image = cv2.flip(image, 1)

            # Set flag
            image.flags.writeable = False

            # Detections
            results = hands.process(image)

            # Set flag to true
            image.flags.writeable = True

            # RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Detections
            print(results)

            # Rendering results
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2,
                                                                     circle_radius=2),
                                              )
            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   # handdetect()
    detect()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
