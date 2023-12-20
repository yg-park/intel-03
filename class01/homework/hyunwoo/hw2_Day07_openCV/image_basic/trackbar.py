import numpy as np
import cv2

cap = cv2.VideoCapture(0)

topLeft = (50, 50)
bold = 0
R=G=B = 0
# Callback function for the trackbar
def on_bold_trackbar(value):
    global bold
    bold = value

def R_trackbar(value):
    
    global R
    R = value

def G_trackbar(value):
    
    global G
    G = value

def B_trackbar(value):
    
    global B
    B = value

cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)
cv2.createTrackbar("R", "Camera", R, 255, R_trackbar)
cv2.createTrackbar("G", "Camera", G, 255, G_trackbar)
cv2.createTrackbar("B", "Camera", B, 255, B_trackbar)

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Text
    cv2.putText(frame, "TEXT",
        topLeft, cv2.FONT_HERSHEY_SIMPLEX, 2, (B, G, R), 1 + bold)

    # Display
    key = cv2.waitKey(1)

    cv2.imshow("Camera",frame)

    if key & 0xFF == ord('q') :
        break
