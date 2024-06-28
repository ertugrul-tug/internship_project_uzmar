import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# Initialize MediaPipe for Hands
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

# Create a visualizer for Open3D
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720, visible=True)

# Load 3D model (.obj)
mesh = o3d.io.read_triangle_mesh("Ship_free.obj")
if not mesh.has_triangles():
    print("Unable to load the model, please check the file path and format.")
    exit()

# Add the mesh to the visualizer
vis.add_geometry(mesh)

# Set up FOV
front = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
vis.get_view_control().set_front(front)
vis.get_view_control().set_up(up)

class VideoProcessingThread(QThread):
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, camera_id, camera_matrix, dist_coeffs, aruco_dict, parameters, marker_size_meters):
        super().__init__()
        self.camera_id = camera_id
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.aruco_dict = aruco_dict
        self.parameters = parameters
        self.marker_size_meters = marker_size_meters
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.prev_rvec_camera = None
        self.prev_tvec_camera = None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

            # Process hand tracking
            results = hands.process(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size_meters, self.camera_matrix, self.dist_coeffs
                )

                for i in range(len(ids)):
                    rvec_marker = rvecs[i][0]
                    tvec_marker = tvecs[i][0]
                    rvec_camera, tvec_camera = self.invert_pose(rvec_marker, tvec_marker)

                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    # Draw the axes for the marker
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec_marker, tvec_marker, self.marker_size_meters)

                    # Calculate the distance between the marker and the camera
                    marker_camera_distance = np.linalg.norm(tvec_camera)

                    # Render the object with the new pose
                    if self.prev_rvec_camera is not None and self.prev_tvec_camera is not None:
                        delta_rvec = rvec_camera - self.prev_rvec_camera
                        delta_tvec = tvec_camera - self.prev_tvec_camera

                        # Convert rotation vector to rotation matrix
                        R, _ = cv2.Rodrigues(delta_rvec)
                        R = R.T  # Transpose for Open3D's rotation format

                        # Transform the object using Open3D methods
                        mesh.rotate(R, center=(0, 0, 0))
                        mesh.translate(delta_tvec.ravel(), relative=True)

                    self.prev_rvec_camera = rvec_camera.copy()
                    self.prev_tvec_camera = tvec_camera.copy()

                    # Update Open3D visualizer
                    vis.update_geometry(mesh)
                    vis.poll_events()
                    vis.update_renderer()

                    # Convert Open3D image to numpy array
                    image = np.asarray(vis.capture_screen_float_buffer(True))

                    # Create a mask from the Open3D image
                    mask = cv2.inRange((image * 255).astype(np.uint8), (1, 1, 1), (255, 255, 255))

                    # Resize mask to match frame size
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                    # Convert Open3D image to BGR format
                    image_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    image_bgr = cv2.resize(image_bgr, (frame.shape[1], frame.shape[0]))

                    # Apply the mask on the frame
                    masked_frame = cv2.bitwise_and(frame, frame, mask=~mask)
                    masked_image_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
                    combined_frame = cv2.add(masked_frame, masked_image_bgr)

                    self.frame_processed.emit(combined_frame)
            else:
                self.frame_processed.emit(frame)

    def invert_pose(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        tvec_inv = -R_inv @ tvec
        rvec_inv, _ = cv2.Rodrigues(R_inv)
        return rvec_inv, tvec_inv

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Load the necessary ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        # Load camera calibration data
        self.camera_matrix = np.load('camera_matrix.npy')
        self.dist_coeffs = np.load('dist_coeffs.npy')

        # Initialize parameters for ArUco marker detection
        self.parameters = cv2.aruco.DetectorParameters()

        # Define the size of the ArUco marker in meters
        self.marker_size_meters = 0.067

        # Set up camera intrinsic parameters for Open3D
        self.width = 1280
        self.height = 720
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height,
                                                           self.camera_matrix[0, 0],
                                                           self.camera_matrix[1, 1],
                                                           self.camera_matrix[0, 2],
                                                           self.camera_matrix[1, 2])

        self.video_thread = VideoProcessingThread(
            camera_id=3,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            aruco_dict=self.aruco_dict,
            parameters=self.parameters,
            marker_size_meters=self.marker_size_meters
        )
        self.video_thread.frame_processed.connect(self.update_display)
        self.video_thread.start()

    def initUI(self):
        self.label = QLabel(self)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.label)
        self.setLayout(self.vbox)

    def update_display(self, frame):
        height, width, channels = frame.shape
        bytesPerLine = channels * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(qImg))

    def closeEvent(self, event):
        self.video_thread.cap.release()
        self.video_thread.quit()
        vis.destroy_window()

def main():
    app = QApplication([])
    ex = VideoWidget()
    ex.show()
    app.exec_()

if __name__ == "__main__":
    main()
