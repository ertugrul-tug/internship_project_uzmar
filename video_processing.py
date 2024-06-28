# video_processing.py

import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
from multiprocessing import Queue, Process

# Initialize MediaPipe for Hands
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

# Load 3D model (.obj)
mesh = o3d.io.read_triangle_mesh("Ship_free.obj")
if not mesh.has_triangles():
    print("Unable to load the model, please check the file path and format.")
    exit()

# Set up Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720, visible=True)
vis.add_geometry(mesh)

# Function to process frames
def process_frames(camera_id, camera_matrix, dist_coeffs, aruco_dict_name, detection_parameters, marker_size_meters, queue):
    # Load the ArUco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_name)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_rvec_camera = None
    prev_tvec_camera = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detection_parameters)

        # Process hand tracking
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size_meters, camera_matrix, dist_coeffs
            )

            for i in range(len(ids)):
                rvec_marker = rvecs[i][0]
                tvec_marker = tvecs[i][0]
                rvec_camera, tvec_camera = invert_pose(rvec_marker, tvec_marker)

                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Draw the axes for the marker
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_marker, tvec_marker, marker_size_meters)

                # Calculate the distance between the marker and the camera
                marker_camera_distance = np.linalg.norm(tvec_camera)

                # Render the object with the new pose
                if prev_rvec_camera is not None and prev_tvec_camera is not None:
                    delta_rvec = rvec_camera - prev_rvec_camera
                    delta_tvec = tvec_camera - prev_tvec_camera

                    # Convert rotation vector to rotation matrix
                    R, _ = cv2.Rodrigues(delta_rvec)
                    R = R.T  # Transpose for Open3D's rotation format

                    # Transform the object using Open3D methods
                    mesh.rotate(R, center=(0, 0, 0))
                    mesh.translate(delta_tvec.ravel(), relative=True)

                prev_rvec_camera = rvec_camera.copy()
                prev_tvec_camera = tvec_camera.copy()

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

                queue.put(combined_frame)
        else:
            queue.put(frame)

    cap.release()
    vis.destroy_window()

def invert_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    tvec_inv = -R_inv @ tvec
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv, tvec_inv

if __name__ == "__main__":
    # Load camera calibration data
    camera_matrix = np.load('camera_matrix.npy')
    dist_coeffs = np.load('dist_coeffs.npy')

    # Initialize parameters for ArUco marker detection
    detection_parameters = cv2.aruco.DetectorParameters()

    # Define the size of the ArUco marker in meters
    marker_size_meters = 0.067

    # Define the ArUco dictionary name
    aruco_dict_name = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Adjust according to your dictionary type

    # Create a queue for inter-process communication
    queue = Queue()

    # Start the video processing in a separate process
    video_process = Process(target=process_frames, args=(3, camera_matrix, dist_coeffs, aruco_dict_name, detection_parameters, marker_size_meters, queue))
    video_process.start()

    video_process.join()  # Wait for the video process to finish
