import cv2
import mediapipe as mp
import time #for FPS

class handDetector():
    def __init__(self, mode= False, maxHands=2, detconf=0.5,trackconf=0.5):
        self.mode=mode
        self.maxHands= maxHands
        self.detconf= detconf
        self.trackconf=trackconf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detconf,self.trackconf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img, handNo=0, draw= True):

        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand= self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id ,lm)
                h, w, c = img.shape  # Height,Width,Channels
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                #if id == 4:  # to see a specific id i.e points on hand out of the 21 points
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return lmList



def main():
    cap = cv2.VideoCapture(0)
    pTime = 0  # Previous Time
    cTime = 0  # Current Time
    detector= handDetector()

    while True:
        success, img = cap.read()
        img= detector.findHands(img)
        lmlist= detector.findPosition(img)
        if len(lmlist) !=0:
            print(lmlist[4]) #for the required index
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()