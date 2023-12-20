import numpy as np
import cv2

cap = cv2.VideoCapture(0)

topLeft = (50,50)

bold = 0
r = 0
g = 0
b = 0

def on_bold_trackbar(value) :
    global bold
    bold = value

def r_trackbar(value) :
    global r
    r = value
def g_trackbar(value) :
    global g
    g = value
def b_trackbar(value) :
    global b
    b = value 


cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)
cv2.createTrackbar("R", "Camera", r, 255, r_trackbar)
cv2.createTrackbar("G", "Camera", g, 255, g_trackbar)
cv2.createTrackbar("B", "Camera", b, 255, b_trackbar)

while(cap.isOpened()) :
    ret, frame = cap.read()

    if ret is False :
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.putText(frame, "FULL", topLeft, cv2.FONT_HERSHEY_SIMPLEX, 2, (b,g,r), 1 + bold)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') :
        break