# HandDetectorModule
import cv2
import mediapipe as mp
import time

class HandsDetector():
    """
    The class for detecting persons hand. The class basics on mediapipe.solutions.hands class
    """
    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mpHands = mp.solutions.hands
        self.hands_detector = self.mpHands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.hand_types = []  # список типов рук, например: ['Left', 'Right']
        self.lmList = []

    def detect_hands(self, img, draw_hands=True):
        """
        Function detects hands and draw their skeletons on the img given to input parameter
        """
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands_detector.process(img_RGB)

        self.hand_types = []

        if self.results.multi_handedness:
            for hand in self.results.multi_handedness:
                label = hand.classification[0].label  # "Left" или "Right"
                # Исправляем направление для зеркального видео
                corrected_label = "Right" if label == "Left" else "Left"
                self.hand_types.append(corrected_label)

        if draw_hands and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img


    def findPositions(self, img, handNumber=0):
        self.lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if handNumber < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[handNumber]
                for id, lm in enumerate(hand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
        return self.lmList

    def get_hand_type(self, handNumber=0):
        if self.hand_types and handNumber < len(self.hand_types):
            return self.hand_types[handNumber]
        return "Unknown"
    
    def fingersAreUp(self, handType):
        """
        Определяет, какие пальцы подняты.
        handType: "Right" или "Left"
        Возвращает список из 5 значений (1 — палец поднят, 0 — опущен)
        """
        fingers = []
        tipIds = []

        if len(self.lmList) == 0:
            return [0, 0, 0, 0, 0]

        tipIds = [4, 8, 12, 16, 20]
        
        if handType == "Right":
            if self.lmList[tipIds[0]][1] < self.lmList[tipIds[0] - 1][1]:
                fingers.append(0)
            else:
                fingers.append(1)
        elif handType == "Left":
            if self.lmList[tipIds[0]][1] > self.lmList[tipIds[0] - 1][1]:
                fingers.append(0)
            else:
                fingers.append(1)
                
        print(self.lmList[4])

        # Проверка остальных 4 пальцев (указательный — мизинец)
        for id in range(1, 5):
            if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


# Testing the module
def main():
    my_detector = HandsDetector()
    cap = cv2.VideoCapture(0)
    ptime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = my_detector.detect_hands(img)
        lmList = my_detector.findPositions(img)
        hand_type = my_detector.get_hand_type()

        if lmList:
            cv2.circle(img, (lmList[8][1], lmList[8][2]), 10, (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f"Hand: {hand_type}", (10, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f"FPS: {int(fps)}", (25, 75),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
