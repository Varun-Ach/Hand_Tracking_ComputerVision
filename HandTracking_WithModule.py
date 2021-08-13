import cv2
import mediapipe as mp
import time #for FPS
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
pTime = 0  # Previous Time
cTime = 0  # Current Time
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img) #Provide argument draw= False ;if u dont the tracking visuals
    lmlist = detector.findPosition(img) #Provide argument draw= False ;if u dont want the circles
    if len(lmlist) != 0:
        print(lmlist[4])  # for the required index
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
