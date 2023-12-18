import numpy as np
import cv2

# Read from the recorded video file
cap = cv2.VideoCapture("ronaldinho.mp4")

# 동영상 파일이 성공적으로 열렸으면 while문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()

    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display
    cv2.imshow("Frame", frame)
    
    # 30 ms동안 대기하며 키 ㅇ입력을 받고 'q' 입력 시 종료 'c' 입력 시 해당 프레임 저장 
    key = cv2.waitKey(30)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('c'):

cap.release()
cv2.destroyAllWindows()


