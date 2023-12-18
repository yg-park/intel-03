import numpy as np
import cv2


# Read from the recorded video file
cap = cv2.VideoCapture("ronaldinho.mp4")

# Get the original video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the new width and height (half of the original size)
new_width = int(width / 2)
new_height = int(height / 2)


# 동영상 파일이 성공적으로 열렸으면 while문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    

    # i ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(33)#33ms만큼 대기 1초가 1000ms->
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('c'):
        #이미지 저장
        break
    
    # Write the resized frame to the output video
    frame = cv2.resize(frame, (new_width, new_height))
    # Display
    cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()
