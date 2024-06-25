import cv2
import numpy as np
import imutils
import time
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

# Initialize MediaPipe Hands
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.label = QLabel(self)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.label)
        self.setLayout(self.vbox)

        # Setup video capture
        self.cap = cv2.VideoCapture(0)  # Change camera index if needed
        self.content = cv2.imread('uzmar.jpg')
        self.content = imutils.resize(self.content, width=600)
        self.imgH, self.imgW = self.content.shape[:2]
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

        self.anchor_marker_id = None
        self.last_seen_marker = None

        self.timer = QTimer(self)
        self.timer.setInterval(20)  # Update at 50 fps
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        arucoParams = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dictionary, parameters=arucoParams)

        if ids is not None:
            if self.anchor_marker_id not in ids.flatten() or self.anchor_marker_id is None:
                self.anchor_marker_id = ids.flatten()[0]
                idx = np.where(ids.flatten() == self.anchor_marker_id)[0][0]
                self.last_seen_marker = corners[idx]

            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if self.last_seen_marker is not None:
                dstMat = np.array([self.last_seen_marker[0][0], self.last_seen_marker[0][1], self.last_seen_marker[0][2], self.last_seen_marker[0][3]])
                srcMat = np.array([[0, 0], [self.imgW, 0], [self.imgW, self.imgH], [0, self.imgH]])

                H, _ = cv2.findHomography(srcMat, dstMat)
                warped = cv2.warpPerspective(self.content, H, (frame.shape[1], frame.shape[0]))

                mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.fillConvexPoly(mask, dstMat.astype("int32"), 255)
                mask_inv = cv2.bitwise_not(mask)
                
                frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                content_fg = cv2.bitwise_and(warped, warped, mask=mask)
                
                frame = cv2.add(frame_bg, content_fg)

        # Process hand tracking
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        # Convert to QImage and display
        height, width, channels = frame.shape
        bytesPerLine = channels * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qImg))

    def closeEvent(self, event):
        self.cap.release()

def main():
    app = QApplication([])
    ex = VideoWidget()
    ex.show()
    app.exec_()

if __name__ == "__main__":
    main()
