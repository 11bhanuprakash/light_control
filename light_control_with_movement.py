import cv2
import mediapipe as mp
import time
import tinytuya

# ================== TUYA BULB SETUP ==================
DEVICE_ID = "bf47d81d1765cce720pchx"
DEVICE_IP = "192.168.178.66"
LOCAL_KEY = "qikv$)D5q3G_IG;h"

bulb = tinytuya.BulbDevice(DEVICE_ID, DEVICE_IP, LOCAL_KEY)
bulb.set_version(3.5)
bulb.turn_on()  # assume light is ON

# ================== HAND DETECTOR ==================
class handDetector():
    def __init__(self, detectionCon=0.7, trackCon=0.7):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            max_num_hands=1,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img):
        lmList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList

# ================== FINGER COUNT ==================
def countFingers(lmList):
    fingers = []

    # Thumb (x direction)
    fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)

    # Other fingers (y direction)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        fingers.append(1 if lmList[tip][2] < lmList[pip][2] else 0)

    return fingers.count(1)

# ================== COLOR CONTROL ==================
def set_color_by_fingers(count):
    colors = {
        1: (255, 0, 0),       # Red
        2: (255, 255, 0),     # Yellow
        3: (0, 255, 0),       # Green
        4: (0, 0, 255),       # Blue
        5: (255, 255, 255)    # White
    }

    if count in colors:
        r, g, b = colors[count]
        bulb.set_colour(r, g, b)
        print(f"🎨 Fingers: {count} → Color set")

# ================== MAIN ==================
def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    prev_count = -1  # prevent repeated commands

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if lmList:
            finger_count = countFingers(lmList)

            if finger_count != prev_count:
                set_color_by_fingers(finger_count)
                prev_count = finger_count

            cv2.putText(
                img, f'Fingers: {finger_count}', (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        cv2.imshow("Hand Controlled Light", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
