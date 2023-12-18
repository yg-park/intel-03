import numpy as np
import cv2


# Read from the first camera device
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

topLeft = (50, 50)
bold = 0
R_color=0
G_color=0
B_color=0
def on_bold_trackbar(value):
   #print("Trackbar value:", value)
   global bold
   bold = value
  
def on_R_color_trackbar(value):
   #print("Trackbar value:", value)
   global R_color
   R_color = value

def on_G_color_trackbar(value):
   #print("Trackbar value:", value)
   global G_color
   G_color = value

def on_B_color_trackbar(value):
   #print("Trackbar value:", value)
   global B_color
   B_color = value


cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)
cv2.createTrackbar("Rcolor", "Camera", bold, 255, on_R_color_trackbar)
cv2.createTrackbar("Gcolor", "Camera", bold, 255, on_G_color_trackbar)
cv2.createTrackbar("Bcolor", "Camera", bold, 255, on_B_color_trackbar)





# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Text
    cv2.putText(frame, "TEXT", topLeft, cv2.FONT_HERSHEY_SIMPLEX, 2, (B_color, G_color, R_color), 1 + bold)

    # Display
    cv2.imshow("Camera",frame)
    # i ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)#33ms만큼 대기 1초가 1000ms->
    if key & 0xFF == ord('q'):
        break
