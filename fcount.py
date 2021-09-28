import numpy

from hd import HandDetector
import cv2
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import numpy as np


def checkOKAY(handLandmarks):
    if handLandmarks[4][1] - handLandmarks[8][1] <= 20 and handLandmarks[4][2] - handLandmarks[8][2] <= 20 \
            and handLandmarks[12][2] < handLandmarks[10][2] and handLandmarks[16][2] < handLandmarks[14][2] \
            and handLandmarks[20][2] < handLandmarks[18][2]:
        return True
    return False


def checkPeace(handLandmarks):
    if handLandmarks[8][2] < handLandmarks[6][2] and handLandmarks[12][2] < handLandmarks[10][2] \
            and handLandmarks[16][2] > handLandmarks[14][2] \
            and handLandmarks[20][2] > handLandmarks[18][2] and handLandmarks[4][1] > handLandmarks[3][1]:
        return True
    return False


def checkRad(handLandmarks):
    if handLandmarks[4][1] < handLandmarks[3][1] and handLandmarks[20][2] < handLandmarks[18][2] \
                and handLandmarks[12][2] > handLandmarks[10][2] and handLandmarks[8][2] > handLandmarks[6][2]\
                and handLandmarks[16][2] > handLandmarks[14][2]:
        return True
    return False



def counter():
    handDetector = HandDetector(min_detection=0.7)
    webcamFeed = cv2.VideoCapture(0)


    while True:
        status, image = webcamFeed.read()
        handLandmarks = handDetector.findHandLandMarks(image=image, draw=False)
        rcount=0
        lcount = 0
        thumbs = "Thumb0"

        if(len(handLandmarks) != 0):
            """
            Brief:
                the first index of the array signifies which point out of the 21 (search up mediapipe hands for reference)
                for every finger besides the thumb we check if the y position of the highest point of a finger is lower than the second lowest point of a finger
                thumb is different because it doesn't go down, more so comes in
            """
            if handLandmarks[4][3] == "Right" and handLandmarks[4][1] > handLandmarks[3][1]:  # Right Thumb
                rcount += 1
            if handLandmarks[4][3] == "Right" and handLandmarks[8][2] < handLandmarks[6][2]:  # Index finger
                rcount += 1
            if handLandmarks[4][3] == "Right" and handLandmarks[12][2] < handLandmarks[10][2]:  # Middle finger
                rcount += 1
            if handLandmarks[4][3] == "Right" and handLandmarks[16][2] < handLandmarks[14][2]:  # Ring finger
                rcount += 1
            if handLandmarks[4][3] == "Right" and handLandmarks[20][2] < handLandmarks[18][2]:  # Little finger
                rcount += 1

            #left
            if handLandmarks[4][3] == "Left" and handLandmarks[4][1] < handLandmarks[3][1]:  # Left Thumb
                 lcount += 1
            if handLandmarks[4][3] == "Left" and handLandmarks[8][2] < handLandmarks[6][2]:  # Index finger
                lcount += 1
            if handLandmarks[4][3] == "Left" and handLandmarks[12][2] < handLandmarks[10][2]:  # Middle finger
                lcount += 1
            if handLandmarks[4][3] == "Left" and handLandmarks[16][2] < handLandmarks[14][2]:  # Ring finger
                lcount += 1
            if handLandmarks[4][3] == "Left" and handLandmarks[20][2] < handLandmarks[18][2]:  # Little finger
                lcount += 1

            #OKAY

            if checkOKAY(handLandmarks):
               cv2.putText(image, "A-OKAY", (10, 375), cv2.FONT_HERSHEY_COMPLEX, 4, (200, 0, 0), 10)
                #peace Sign
            if checkPeace(handLandmarks):
                cv2.putText(image, "Peace", (10, 375), cv2.FONT_HERSHEY_COMPLEX, 4, (200, 0, 0), 10)
                #rad
            if checkRad(handLandmarks):
                cv2.putText(image, "RAD!!", (10, 375), cv2.FONT_HERSHEY_COMPLEX, 4, (200, 0, 0), 10)





       # cv2.putText(image, str(rcount), (80, 375), cv2.FONT_HERSHEY_COMPLEX, 4, (200, 0, 0), 10)
        #cv2.putText(image, str(lcount+rcount), (10, 375), cv2.FONT_HERSHEY_COMPLEX, 4, (200, 0, 0), 10)
        cv2.imshow("Hands", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break



def audiocontrol(volume):
    setvol = 0.6525*(volume)-65.25
    # Get default audio device using PyCAW
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevel(setvol, None)
    currentVolumeDb = volume.GetMasterVolumeLevel()


def volcontrol():
    handDetector = HandDetector(min_detection=0.7)
    webcamFeed = cv2.VideoCapture(0)
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    while True:
        status, image = webcamFeed.read()
        handLandmarks = handDetector.findHandLandMarks(image=image, draw=True)
        currentVolumeDb = volume.GetMasterVolumeLevel()
        if (len(handLandmarks) != 0):
            #print(handLandmarks[0][3])
            #qqqqqqprint("number",handLandmarks[8][1] - handLandmarks[4][1],"kuler" ,handLandmarks[4][2] - handLandmarks[8][2])
            if handLandmarks[0][3] == "Left" and handLandmarks[4][1] - handLandmarks[8][1] <= 20 and handLandmarks[4][2] - handLandmarks[8][2] <= 20:
               # print("yo")
                currentVolumeDb -= 3
                if currentVolumeDb <= -65.25:
                    currentVolumeDb = -65.25

            elif handLandmarks[0][3] == "Left" and handLandmarks[8][1] - handLandmarks[4][1] >= 120 or handLandmarks[4][2] - handLandmarks[8][2] >= 80:
               # print("ere")
                currentVolumeDb += 3
                if currentVolumeDb >= 0:
                    currentVolumeDb = 0
        volume.SetMasterVolumeLevel(currentVolumeDb, None)
        cv2.imshow("Hands", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break




if __name__ == '__main__':
    #counter()
    volcontrol()
