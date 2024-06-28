# video_widget.py

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from multiprocessing import Queue
from video_processing import queue  # Import the queue from video_processing

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.label = QLabel(self)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.label)
        self.setLayout(self.vbox)

        # Start a timer to update the display
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(20)  # Update every 20 milliseconds

    def update_display(self):
        if not queue.empty():
            frame = queue.get()

            height, width, channels = frame.shape
            bytesPerLine = channels * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_BGR888)
            self.label.setPixmap(QPixmap.fromImage(qImg))

    def closeEvent(self, event):
        sys.exit()

def main():
    app = QApplication(sys.argv)
    ex = VideoWidget()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
