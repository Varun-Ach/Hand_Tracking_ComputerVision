import cv2
import mediapipe as mp
import time #for FPS

cap= cv2.VideoCapture(0)
mpHands= mp.solutions.hands
hands= mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
pTime=0 #Previous Time
cTime=0 #Current Time

while True:
    success, img= cap.read()
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results= hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id , lm in enumerate(handlms.landmark):
                #print(id ,lm)
                h,w,c= img.shape #Height,Width,Channels
                cx,cy= int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id==4: #to see a specific id i.e points on hand out of the 21 points
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)


            mpDraw.draw_landmarks(img,handlms, mpHands.HAND_CONNECTIONS)

    cTime=time.time()
    fps= 1/(cTime-pTime)
    pTime= cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)