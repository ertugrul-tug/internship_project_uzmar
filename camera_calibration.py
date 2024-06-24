import cv2
import numpy as np

# Load the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Marker size in meters (adjust based on your marker size)
marker_length = 0.67

frame_count = 0

# Object points in real world space for a single marker
objp = np.array([[0, 0, 0], [marker_length, 0, 0], [marker_length, marker_length, 0], [0, marker_length, 0]], dtype=np.float32)

# Video file path
video_path = 'video.mkv'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Lists to store object points and image points for all frames
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count = frame_count + 1

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for i in range(len(ids)):
            imgpoints.append(corners[i])
            objpoints.append(objp)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert lists to numpy arrays
objpoints = np.array(objpoints, dtype=np.float32)
imgpoints = np.array(imgpoints, dtype=np.float32)



# Check if we have enough points for calibration
if len(imgpoints) >= 10:  # Ensure at least 10 points
    
    print("ife girdi")
    print("Frame sayısı:\n", frame_count)
    # Calibrate the camera
    try:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("hesaplamaları yaptı")
        
        # Save the calibration data
        np.save('camera_matrix.npy', camera_matrix)
        np.save('dist_coeffs.npy', dist_coeffs)

        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)

    except cv2.error as e:
        print(f"Calibration failed: {e}")
        camera_matrix = None
        dist_coeffs = None

else:
    print("Not enough points for calibration. Please capture more frames with ArUco markers.")
