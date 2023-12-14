import numpy as np
import cv2

bold = 0
fSize = 0
R = 0
G = 255
B = 255

# Callback function for the trackbar
def on_bold_trackbar(value):
    global bold
    bold = value

def on_fSize_trackbar(value):
    global fSize
    fSize = value

def on_R_trackbar(value):
    global R
    R = value

def on_G_trackbar(value):
    global G
    G = value

def on_B_trackbar(value):
    global B
    B = value

cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)
cv2.createTrackbar("fSize", "Camera", fSize, 10, on_fSize_trackbar)
cv2.createTrackbar("R", "Camera", R, 255, on_R_trackbar)
cv2.createTrackbar("G", "Camera", G, 255, on_G_trackbar)
cv2.createTrackbar("B", "Camera", B, 255, on_B_trackbar)

cap = cv2.VideoCapture(0)
topLeft = (50, 50)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    cv2.putText(frame, "TEXT",
                topLeft, cv2.FONT_HERSHEY_SIMPLEX, 1 + fSize, (R, G, B), 1 + bold)
    
    cv2.imshow("Camera",frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

