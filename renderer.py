import numpy as np
import argparse
import imutils
import sys
import cv2
import cv2.aruco as aruco

# Load the camera
cap = cv2.VideoCapture(0)
content  = cv2.imread('uzmar.jpg')
content = imutils.resize(content, width=600)
(imgH, imgW) = content.shape[:2]

# Set the dictionary to use
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
'''
num_markers_x = 4
num_markers_y = 4
marker_length = 0.04
marker_separation = 0.01



board = cv2.aruco.GridBoard(
    num_markers_x,              # Number of markers in the X direction.
    num_markers_y,              # Number of markers in the Y direction.
    marker_length,              # Length of the marker side.
    marker_separation,          # Length of the marker separation.
    dictionary                  # The dictionary of the markers.
                                # (optional) Ids of all the markers (X*Y markers).
)

img = board.draw( 
    content.shape,                  # Size of the output image in pixels.
    content,                        # Output image with the board
    0,                          # Minimum margins (in pixels) of the board in the output image
    1                           # Width of the marker borders
)
extension = ".jpg"
'''

'''
# constructing of parser for our arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="pat to input image containing ArUCo tag")
ap.add_argument("-s","--source", required=True, help="path to input source image that will be put on input")
args = vars(ap.parse_args())
'''
while(True):
    ret, frame = cap.read()
    
    # Detect ArUco markers
    arucoParams = cv2.aruco.DetectorParameters()
    (corners, ids, _) = cv2.aruco.detectMarkers(frame, dictionary, parameters=arucoParams)
    
    if ids is not None:
        #for i in range(ids.size):
            # Estimate marker pose
            #rvec, tvec = aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
            
            # Project virtual content onto the marker
            # ...
            
        # Draw markers
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
    
    cv2.imshow("app window", frame)
    '''
    # Capture frame-by-frame
    ret, frame = cap.read()

    
    arucoParams = cv2.aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, dictionary, parameters=arucoParams)

    # initializing our list of reference points
    print("[INFO] contructing augmented reality visulization...")
    refPts = []
    # looping ArUCo markers
    does_corners_exist = False
    for i in corners:
        # grabbing and appending our list of reference points
        j = np.where(ids == i)
        corner = corners[i]
        refPts.append(corner)
        does_corners_exist = True
            
    # unpacking the reference points coordinates
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)

    # Draw markers
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # defining transform matrix of our source image
    (srcH, srcW) = frame.shape[:2]
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

    # computing homography and warping our source image
    (H, _) = cv2.findHomography(srcMat, dstMat)
    warped = cv2.warpPerspective(frame, H, (imgW, imgH))

    # constructing the mask of the warped image
    mask = np.zeros((imgH, imgW), dtype="uint8")
    cv2.filConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv2.LINE_AA)

    # enable if you want rendered warped image to have a black border around it
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, rect, iterations=2)

    # making copy of our mask in order to put our image into it
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)

    # multiplying masked and unmasked pixels with our image then adding them together
    warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    imageMultiplied = cv2.multiply(content.astype(float), 1.0 - maskScaled)
    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('output', output)
    '''

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
# loading input image for our usage
print("[INFO] loading input image and source image")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(imgH, imgW) = image.shape[:2]

# loading source image
source = cv2.imread(args["source"])

# load the ArUCo for image detection
print("[INFO] detecting markers")
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

# failure to find image markers
if len(corners) != 4:
    print("[INFO] could not find all 4 corners...exiting")
    sys.exit(0)

# initializing our list of reference points
print("[INFO] contructing augmented reality visulization...")
ids = ids.flatten()
refPts = [0,0,0,0]

# looping ArUCo markers

for i in (923, 1001, 241, 1007):

    # grabbing and appending our list of reference points
    j= np.squeeze(np.where(ids== i))
    corner = np.squeeze(corners[j])
    refPts[i] = corner

    # unpacking the reference points coordinates
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)

# defining transform matrix of our source image
(srcH, srcW) = source.shape[:2]
srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

# computing homography and warping our source image
(H, _) = cv2.findHomography(srcMat, dstMat)
warped = cv2.warpPerspective(source, H, (imgW, imgH))

# constructing the mask of the warped image
mask = np.zeros((imgH, imgW), dtype="uint8")
cv2.filConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv2.LINE_AA)

# enable if you want rendered warped image to have a black border around it
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.dilate(mask, rect, iterations=2)

# making copy of our mask in order to put our image into it
maskScaled = mask.copy() / 255.0
maskScaled = np.dstack([maskScaled] * 3)

# multiplying masked and unmasked pixels with our image then adding them together
warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
imageMultiplied = cv2.multiply(image.astype(float), 1.0 - maskScaled)
output = cv2.add(warpedMultiplied, imageMultiplied)
output = output.astype("uint8")

# rendering everything
cv2.imshow("Input", image)
cv2.imshow("Source", source)
cv2.imshow("OpenCV AR Output", output)
cv2.waitKey(0)
'''