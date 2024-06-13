import numpy as np
import cv2

# Define the variables
output_image_path = "output_aruco.jpg"
aruco_id = 0  # ID of ArUCo tag to generate
aruco_type = "DICT_6X6_50"  # Type of ArUCo tag to generate

# Define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# Verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(aruco_type, None) is None:
    print(f"[INFO] ArUCo tag of '{aruco_type}' is not supported")
    exit()

# Load the ArUCo dictionary
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

# Allocate memory for the output ArUCo tag and then draw the ArUCo tag on the output image
print(f"[INFO] generating ArUCo tag type '{aruco_type}' with ID '{aruco_id}'")
tag = np.zeros((300, 300, 1), dtype="uint8")
cv2.aruco.generateImageMarker(arucoDict, aruco_id, 300, tag, 1)

# Write the generated ArUCo tag to disk
cv2.imwrite(output_image_path, tag)

# Display the generated ArUCo tag
cv2.imshow("ArUCo Tag", tag)
cv2.waitKey(0)
