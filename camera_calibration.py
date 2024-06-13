import cv2
import numpy as np
import os

# Load the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters()

# Marker size in meters (adjust based on your marker size)
marker_length = 0.05

# Arrays to store object points and image points from all images
all_corners = []
all_ids = []

# Object points in real world space for a single marker
objp = np.array([[0, 0, 0], [marker_length, 0, 0], [marker_length, marker_length, 0], [0, marker_length, 0]], dtype=np.float32)

# Directory containing calibration images
calibration_dir = 'calibration_images'  # Replace with the actual path to your images
images = [os.path.join(calibration_dir, fname) for fname in os.listdir(calibration_dir) if fname.endswith('.jpg')]

# Lists to store object points and image points for all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

for image_path in images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for corner in corners:
            imgpoints.append(corner)
            objpoints.append(objp)

# Convert lists to numpy arrays
objpoints = np.array(objpoints, dtype=np.float32)
imgpoints = np.array(imgpoints, dtype=np.float32)

# Check if we have enough points for calibration
if len(imgpoints) >= 10:  # Ensure at least 10 points
    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the calibration data
    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)

    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
else:
    print("Not enough points for calibration. Please capture more calibration images.")
