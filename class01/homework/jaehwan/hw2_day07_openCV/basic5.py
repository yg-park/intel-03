"""
1. 가지고 있는 카메라의 지원 가능한 해상도를 확인 후 해상도 변경
2. 카메라 input을 output.mp4 동영상 파일로 저장하도록 코드 추가
"""
import cv2
#read from the first camera device
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

w=320#1280#1920
h=240#720#1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH,w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)

#성공적으로 video device가 열렸으면 while 문 반복
while(cap.isOpened()):
    #한 프레임을 읽어옴
    ret, frame=cap.read()
    if ret is False:
        print("can't receive frame (stream end?). exiting...")
        break
    #display
    cv2.imshow("camera",frame)
    #1ms동안 대기하며 키 입력을 받고 'q'입력시 종료
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
         break
