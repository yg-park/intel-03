import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

topLeft = (50, 50)
topLeft2 = (50, 100)
topLeft3 = (50, 150)
bold = 0
size = 0
color = 0

# Callback function for the trackbar
def on_bold_trackbar(value):
    #print("Trackbar value:", value)
    global bold
    bold = value

# Callback function for the trackbar
def on_size_trackbar(value):
    #print("Trackbar value:", value)
    global size
    size = value

# Callback function for the trackbar
def on_color_trackbar(value):
    #print("Trackbar value:", value)
    global color
    color = value

cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)
cv2.createTrackbar("size", "Camera", bold, 10, on_size_trackbar)
cv2.createTrackbar("color", "Camera", bold, 10, on_color_trackbar)

# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()

    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Text
    cv2.putText(frame, "TEXT", topLeft, cv2.FONT_HERSHEY_SIMPLEX, 2, (0+bold*10, 255, 255), 1 + bold)
    cv2.putText(frame, "TEXT_Size", topLeft2, cv2.FONT_HERSHEY_SIMPLEX, 2*(size+1), (0, 255, 255), 5)
    cv2.putText(frame, "TEXT_Color", topLeft3, cv2.FONT_HERSHEY_SIMPLEX, 2, ((color+1)*25, 125, 255), 5)

    # Display
    cv2.imshow("Camera",frame)

    key=cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


