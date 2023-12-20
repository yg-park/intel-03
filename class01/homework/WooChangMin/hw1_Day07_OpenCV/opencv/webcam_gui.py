import numpy as np
import cv2


# Read from the first camera device
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

topLeft = (50,50)
bottomRight = (300,300)


# 성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Display
    cv2.imshow("Camera", frame)
    # Line
    cv2.line(frame, topLeft, bottomRight, (0,255,0), 5)

    # Rectangle
    cv2.rectangle(frame, [pt+30 for pt in topLeft], [pt-30 for pt in bottomRight], (0,0,255), 5)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'me', [pt+30 for pt in topLeft], font, 2, (0,255, 255), 10)
    # i ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)#33ms만큼 대기 1초가 1000ms->
    if key & 0xFF == ord('q'):
        break
