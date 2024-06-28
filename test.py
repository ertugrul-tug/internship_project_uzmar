import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

# Initialize MediaPipe for Hands
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Load the neccesary ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        # Load camera calibration data
        self.camera_matrix = np.load('camera_matrix.npy')
        self.dist_coeffs = np.load('dist_coeffs.npy')

        # Initialize parameters for ArUco marker detection
        self.parameters = cv2.aruco.DetectorParameters()

        # Define the size of the ArUco marker in meters
        self.marker_size_meters = 0.067

        # Load 3D model (.obj)
        self.mesh = o3d.io.read_triangle_mesh("Ship_free.obj")
        if not self.mesh.has_triangles():
            print("Unable to load the model, please check the file path and format.")
            exit()
        
        # Create a visualizer for Open3D
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=1280, height=720, visible=False)

        # Add the mesh to the visualizer
        self.vis.add_geometry(self.mesh)

        # Set up FOV
        front = [1.0, 0.0, 0.0]
        up = [0.0, 1.0, 0.0]
        self.vis.get_view_control().set_front(front)
        self.vis.get_view_control().set_up(up)

        # Set up camera intrinsic parameters for Open3D
        self.width = 1280
        self.height = 720
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height,
                                                           self.camera_matrix[0,0],
                                                           self.camera_matrix[1,1],
                                                           self.camera_matrix[0,2],
                                                           self.camera_matrix[1,2])
        
        self.prev_rvec_camera = None
        self.prev_tvec_camera = None

    def initUI(self):
        self.label = QLabel(self)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.label)
        self.setLayout(self.vbox)

        # Setup video capture
        self.cap = cv2.VideoCapture(3)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size_meters, self.camera_matrix, self.dist_coeffs
            )

            for i in range(len(ids)):
                rvec_marker = rvecs[i][0]
                tvec_marker = tvecs[i][0]
                rvec_camera, tvec_camera = self.invert_pose(rvec_marker, tvec_marker)

                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Calculate the distance between the marker and the camera
                marker_camera_distance = np.linalg.norm(tvec_camera)

            # Render the object with the new pose
            if self.prev_rvec_camera is not None and self.prev_tvec_camera is not None:
                delta_rvec = rvec_camera - self.prev_rvec_camera
                delta_tvec = tvec_camera - self.prev_tvec_camera

                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(delta_rvec)
                R = R.T # Transpose for Open3D's rotation format

                # Round the rotation matrix to 2 decimals
                R_rounded = np.round(R, decimals=2)

                # Transfrom the object using Open3D methods
                self.mesh.rotate(R_rounded, center=(0, 0, 0))

                # Calculate the scale factor based on the marker-camera distance
                scale_factor = marker_camera_distance * 0.1 # Adjust the scaling factor as needed
                self.mesh.scale(scale_factor, center=(0, 0, 0))

                # Anchor the object to the marker
                self.mesh.translate(delta_tvec.ravel(), realtive=True)

            self.prev_rvec_camera = rvec_camera.copy()
            self.prev_tvec_camera = tvec_camera.copy()

            # Process hand tracking
            results = hands.process(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Update Open3D visualizer
            self.vis.update_geometry(self.mesh)
            self.vis.poll_events()
            self.vis.update_renderer()

            # Convert Open3D image to numpy array
            image = np.asarray(self.vis.capture_screen_float_buffer(True))

            # Create a mask from the Open3d İMAGE
            mask = cv2.inRange(image, (100, 100, 100), (255, 255, 255))

            # Apply the mask on the frame
            masked_frame = cv2.bitwise_and(frame,frame, mask=mask)

            # Convert the masked frame to QImage and display
            height, width, channels = masked_frame.shape
            bytesPerLine = channels * width
            qImg = QImage(
                masked_frame.data, width, height, bytesPerLine, QImage.Format_BGR888
            )
            self.label.setPixmap(QPixmap.fromImage(qImg))

    def closeEvent(self, event):
        self.cap.release()
        self.vis.destor_window()

    def invert_pose(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        tvec_inv = -R_inv @ tvec
        rvec_inv, _ = cv2.Rodrigues(R_inv)
        return rvec_inv, tvec_inv
    
def main():
    app = QApplication([])
    ex = VideoWidget()
    ex.show()
    app.exec_()

if __name__ == "__main__":
    main()