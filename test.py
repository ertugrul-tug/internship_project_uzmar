import cv2
import numpy as np
import open3d as o3d

# Load the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Load camera calibration data
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

# Initialize parameters for ArUco marker detection
parameters = cv2.aruco.DetectorParameters()

# Define the size of the ArUco marker in meters
marker_size_meters = 0.0067  # 6.7cm converted to meters

# Load your 3D model (converted to OBJ)
mesh = o3d.io.read_triangle_mesh("Ship_free.obj")
if not mesh.has_triangles():
    print("Unable to load the model, please check the file path and format.")
    exit()

# Create a visualizer for Open3D
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the mesh to the visualizer
vis.add_geometry(mesh)

# Set up camera intrinsic parameters for Open3D
width = 640
height = 480
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])

def render_object(vis, mesh, intrinsic, extrinsic):
    # Create PinholeCameraParameters object
    parameters = o3d.camera.PinholeCameraParameters()
    parameters.intrinsic = intrinsic
    parameters.extrinsic = extrinsic  # Assign the extrinsic matrix directly

    # Clear and update the visualizer with the new intrinsic and extrinsic parameters
    vis.clear_geometries()
    vis.add_geometry(mesh)
    vis.get_view_control().convert_from_pinhole_camera_parameters(parameters)
    vis.poll_events()
    vis.update_renderer()

    # Capture the screen image from Open3D and convert to numpy array
    img = np.asarray(vis.capture_screen_float_buffer(True))
    img = (img * 255).astype(np.uint8)  # Convert to 8-bit integer for OpenCV compatibility
    return img

# Start capturing video input from the webcam
cap = cv2.VideoCapture(3)  # Adjust camera index as needed

def invert_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    tvec_inv = -R_inv @ tvec
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv, tvec_inv

def draw_text(img, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 0)
    thickness = 1
    cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

# Initialize variables for object pose update
prev_rvec_camera = None
prev_tvec_camera = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size_meters, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            rvec_marker = rvecs[i][0]
            tvec_marker = tvecs[i][0]
            rvec_camera, tvec_camera = invert_pose(rvec_marker, tvec_marker)

            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Render the object with the new pose
            if prev_rvec_camera is not None and prev_tvec_camera is not None:
                delta_rvec = rvec_camera - prev_rvec_camera
                delta_tvec = tvec_camera - prev_tvec_camera

                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(delta_rvec)
                R = R.T  # Transpose for Open3D's rotation format

                # Round the rotation matrix to 2 decimals
                R_rounded = np.round(R, decimals=3)

                mesh.rotate(R_rounded, center=(0, 0, 0))
                mesh.translate(delta_tvec.ravel(), relative=True)

            prev_rvec_camera = rvec_camera.copy()
            prev_tvec_camera = tvec_camera.copy()

            draw_text(frame, f"Rvec: {rvec_camera.flatten()}", (10, 30))
            draw_text(frame, f"Tvec: {tvec_camera.flatten()}", (10, 50))
            draw_text(frame, f"Distance: {np.linalg.norm(tvec_camera):.2f}m", (10, 70))

    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    cv2.imshow('Augmented Reality', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()