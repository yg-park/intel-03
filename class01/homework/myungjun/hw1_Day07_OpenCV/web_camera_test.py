import cv2
import numpy as np

# Read from the first camera device
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

w = 640 #1280#1920
h = 480 #720#1080

topLeft = (50, 50)
bottomRight = (300, 300)
bold = 0
tSize = 0
cR = 0
cG = 0
cB = 0

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


#Callback function for the trackbar
def on_bold_trackbar(value):
    #print("Trackbar value:", value)
    global bold
    bold = value


#Callback function for the trackbar
def on_size_trackbar(value):
    #print("Trackbar value:", value)
    global tSize
    tSize = value


#Callback function for the trackbar
def on_cR_trackbar(value):
    #print("Trackbar value:", value)
    global cR
    cR = value


#Callback function for the trackbar
def on_cG_trackbar(value):
    #print("Trackbar value:", value)
    global cG
    cG = value


#Callback function for the trackbar
def on_cB_trackbar(value):
    #print("Trackbar value:", value)
    global cB
    cB = value


cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)
cv2.createTrackbar("tSize", "Camera", tSize, 10, on_size_trackbar)
cv2.createTrackbar("cR", "Camera", cR, 255, on_cR_trackbar)
cv2.createTrackbar("cG", "Camera", cG, 255, on_cG_trackbar)
cv2.createTrackbar("cB", "Camera", cB, 255, on_cB_trackbar)


# If video device was successfully opened
while(cap.isOpened()):
    # Read one frame
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Line
    cv2.line(frame, topLeft, bottomRight, (0, 255, 0), 5)

    # Rectangle
    cv2.rectangle(frame, [pt+30 for pt in topLeft], [pt-30 for pt in bottomRight], (0, 0, 255), 5)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'YOU', [pt+80 for pt in topLeft], font, 1 + tSize, (cR, cG, cB), 1 + bold)

    # Display
    cv2.imshow("Camera", frame)

    # Wait 1ms for key entry and terminate if 'q' is entered
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()