import numpy as np
import cv2

# Read from the recorded video file
cap = cv2.VideoCapture(0, cv2.CAP_V4L)

# Get the original video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# 동영상 파일이 성공적으로 열렸으면 while문 반복
while cap.isOpened():
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display
    cv2.imshow("Camera", frame)
    
    # Write the frame into the output video file
    out.write(frame)
    
    # i ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
