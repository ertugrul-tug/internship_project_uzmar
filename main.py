import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
# Now configured to track multiple hands (up to 4)
hands = mp_hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

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
        self.cap = cv2.VideoCapture(3)  # Change camera index if needed
        self.timer = QTimer(self)
        self.timer.setInterval(20)  # Update at 50 fps
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                       mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2))

        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.cap.release()

def main():
    app = QApplication([])
    ex = VideoWidget()
    ex.show()
    app.exec_()

if __name__ == "__main__":
    main()
