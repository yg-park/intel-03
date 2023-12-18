"""
1.동영상이 빠르게 재생된다 정상으로 수정
2.동영상이 끝까지 재생되면 반복하게 수정
3.동영상 크기를 반으로 resize해서 출력
4.동영상 재생중 c키 입력받으면 해당 프레임을 이미지 파일로 저장하게 코드 수정
이름은 001.jpg 002.jpg등으로 overwrite되지않게 하자.
"""
import numpy as np
import cv2

#read from the recorded video file
cap=cv2.VideoCapture("ronaldinho.mp4")

#동영상 파일이 성공적으로 열렸으면 while 문 반복

while(cap.isOpened()):
    #한 프레임을 읽어옴
    ret, frame=cap.read()
    
    if ret is False:
        #print("Can't receive frame (stream end?). Exiting...")
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue
        #break
        

    # display
    cv2.imshow("Frame",frame)
    #1ms동안 대기하며 키 입력을 받고 q입력시 종료
    key = cv2.waitKey(30)
    if key & 0xFF==ord('q'):
         break

cap.release()
cv2.destroyAllWindows()
