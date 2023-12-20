import numpy as np
import cv2

# Read from the recorded video file
cap = cv2.VideoCapture("ronaldinho.mp4")

frame_count = 0

# 동영상 파일이 성공적으로 열렸으면 while문 반복
while(cap.isOpened()):
    # 한 프레임을 읽어옴
    ret, frame = cap.read()
    
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Resize to half
    height, width = frame.shape[:2]
    resized_frame = cv2.resize(frame, (int(width / 2), int(height / 2)))
    
    # Display
    cv2.imshow("Frame", resized_frame)
    
    # 33 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(33)
    
    if key & 0xFF == ord('c'):
        frame_count += 1
        cv2.imwrite(f"{frame_count:03d}.jpg", resized_frame)
        print(f"Frame {frame_count:03d}.jpg saved")
        
    if key & 0xFF == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()

"""
# 한 번 영상이 끝나면 처음부터 다시 시작
while True:
    # 다음 프레임으로 이동
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Restarting the video ...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 비디오의 프레임 인덱스를 처음(0)으로 설정
        continue

    # Display
    cv2.imshow("Frame", frame)

    # 33 ms 동안 대기하며 키 입력을 받고 'q' 입력 시 종료
    key = cv2.waitKey(33)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""