import numpy as np
import cv2
import cv2.aruco as aruco
import imutils

def main():
    camera()

def camera():
    # Load the camera
    cap = cv2.VideoCapture(3)  # Change to the appropriate camera index if needed
    content = cv2.imread('uzmar.jpg')
    content = imutils.resize(content, width=600)
    (imgH, imgW) = content.shape[:2]
    
    # Set the dictionary to use
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

    while True:
        ret, feed = cap.read()

        # Resize the frame to 1280x720
        #feed = cv2.resize(feed, (1280, 720))

        frame = feed.copy()

        
        cv2.imshow("feed", feed)

        if not ret:
            print("Failed to grab frame")
            break
        
        # Detect ArUco markers
        arucoParams = aruco.DetectorParameters()  # Corrected this line
        corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=arucoParams)

        if ids is not None:
            
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            corners_data = [[ids[i][0], corners[i]] for i in range(len(ids))]

            if len(corners) == 4:

                # Get corner data and assign them accordingly
                refPts = [None] * 4
                for i in range(4):
                    if corners_data[i][0] == 0:
                        refPts[0] = corners_data[i][1][0][3]
                    elif corners_data[i][0] == 1:
                        refPts[1] = corners_data[i][1][0][2]
                    elif corners_data[i][0] == 2:
                        refPts[2] = corners_data[i][1][0][0]
                    elif corners_data[i][0] == 3:
                        refPts[3] = corners_data[i][1][0][1]

                dstMat = np.array([refPts[2], refPts[3], refPts[1], refPts[0]])

                # Define transform matrix of our source image
                srcMat = np.array([[0, 0], [imgW, 0], [imgW, imgH], [0, imgH]])

                # Compute homography and warp our content image
                H, _ = cv2.findHomography(srcMat, dstMat)
                warped = cv2.warpPerspective(content, H, (feed.shape[1], feed.shape[0]))

                # Construct the mask of the warped image
                mask = np.zeros(feed.shape[:2], dtype="uint8")
                cv2.fillConvexPoly(mask, dstMat.astype("int32"), 255, cv2.LINE_AA)

                # Invert the mask
                mask_inv = cv2.bitwise_not(mask)

                # Use the mask to extract the region of interest from the feed
                feed_bg = cv2.bitwise_and(feed, feed, mask=mask_inv)

                # Use the inverted mask to extract the warped content image
                content_fg = cv2.bitwise_and(warped, warped, mask=mask)

                # Combine the feed background and content foreground
                output = cv2.add(feed_bg, content_fg)

                cv2.imshow("Output", output)
                cv2.imshow("frame", frame)
            else:
                cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()