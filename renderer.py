import numpy as np
import cv2
import cv2.aruco as aruco
import imutils
import time
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils

def main():
    camera()

def camera():
    # Load the camera
    cap = cv2.VideoCapture(3)  # Change to the appropriate camera index if needed
    if not cap.isOpened():
        print("Failed to open camera.")
        return
    
    content = cv2.imread('uzmar.jpg')
    content = imutils.resize(content, width=600)
    (imgH, imgW) = content.shape[:2]
    
    # Set the dictionary to use for ArUco markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50) #DICT_5X5_250

    # Setup MediaPipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)

    # Anchor point and image initial setup
    anchor_marker_id = None
    anchor_image = None
    anchor_time = None
    last_seen_marker = None

    while True:
        ret, feed = cap.read()

        if not ret:
            print("Failed to grab frame")
            break
        
        frame = feed.copy()

        # Detect ArUco markers
        arucoParams = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=arucoParams)

        if ids is not None:
            # Reset anchor if a new ArUco marker is detected or previously lost marker is found
            if anchor_marker_id not in ids.flatten() or anchor_marker_id is None:
                anchor_marker_id = ids.flatten()[0]
                anchor_image = content.copy()
                anchor_time = time.time()
                idx = np.where(ids.flatten() == anchor_marker_id)[0][0]
                last_seen_marker = corners[idx]

            # Draw detected markers
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Display anchored image if anchor exists
            marker_corner = last_seen_marker[0]
            dstMat = np.array([marker_corner[0], marker_corner[1], marker_corner[2], marker_corner[3]])

            # Define source matrix based on the image dimensions
            srcMat = np.array([[0, 0], [imgW, 0], [imgW, imgH], [0, imgH]])

            # Compute homography and warp the content image if anchor exists
            H, _ = cv2.findHomography(srcMat, dstMat)
            warped = cv2.warpPerspective(anchor_image, H, (frame.shape[1], frame.shape[0]))

            # Construct the mask of the warped image
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.fillConvexPoly(mask, dstMat.astype("int32"), 255, cv2.LINE_AA)

            # Invert the mask
            mask_inv = cv2.bitwise_not(mask)

            # Use the mask to extract the region of interest from the feed
            feed_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

            # Use the inverted mask to extract the warped content image
            content_fg = cv2.bitwise_and(warped, warped, mask=mask)

            # Combine the feed background and content foreground
            frame = cv2.add(feed_bg, content_fg)

            # Display the anchored image
            cv2.imshow("Anchored Image", frame)

        # Detect hands in the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Process hand landmarks and gestures
        pinch_detected = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                # Implement pinch gesture logic here
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Transform coordinates to match the warped image if anchor exists
                if anchor_image is not None:
                    h, w, c = warped.shape
                    thumb_coords = (int(thumb.x * w), int(thumb.y * h))
                    index_coords = (int(index.x * w), int(index.y * h))

                    # Check if the pinch is within the mask
                    if cv2.pointPolygonTest(dstMat.astype("int32"), thumb_coords, False) >= 0 and cv2.pointPolygonTest(dstMat.astype("int32"), index_coords, False) >= 0:
                        pinch_distance = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
                        if pinch_distance < 0.05:  # Adjust threshold based on hand scale
                            pinch_detected = True
                            # Perform image manipulation based on pinch gesture
                            print("Pinch detected")
                            break

        # Reset anchor if pinch gesture is not detected
        if not pinch_detected:
            anchor_marker_id = None
            anchor_image = None
            anchor_time = None
            #if cv2.getWindowProperty("Anchored Image", cv2.WND_PROP_VISIBLE) > 0:
                #cv2.destroyWindow("Anchored Image")

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
