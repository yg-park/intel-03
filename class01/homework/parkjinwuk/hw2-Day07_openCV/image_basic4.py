import numpy as np
import cv2
import time

# Read from the recorded video file
cap = cv2.VideoCapture("ronaldinho.mp4")

# 동영상 파일이 성공적으로 열렸으면 while문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    time.sleep(0.05)
    
    frame = cv2.resize(frame,dsize=None, fx=0.5, fy=0.5)
    
    if(cap.get(cv2.CAP_PROP_POS_FRAMES)==cap.get(cv2.CAP_PROP_FRAME_COUNT)):
    	cap.open("ronaldinho.mp4")
    
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display
    cv2.imshow("Frame",frame)

    # 1ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 1. 동영상이 너무 빠르게 재생된다. 이유를 찾아보고 정상적인 속도로 재생될 수 있도록 수정해보자.
# 2. 동영상이 끝까지 재생되면 더이상 frame을 읽어오지 못해 종료된다. 동영상이 끝까지 재생되면 다시 처음부터 반복 될 수 있도록 수정해보자.
# 3. 동영상 크기를 반으로 resize 해서 출력해보자.
# 4. 동영상 재생 중 'c'키 입력을 받으면 해당 프레임을 이미지 파일로 저장하게 코드를 수정해보자. 파일 이름은 001.jpg, 002.jpg 등으로 overwrite되지 않게 하자.
