import numpy as np
import cv2

drawing = False  # 마우스 클릭 상태 확인을 위한 플래그
circle_center = (300, 100)  # 원의 중심 초기값
radius = 50  # 원의 반지름 초기값

def draw_circle(event, x, y, flags, param):
    global circle_center, drawing, radius

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼을 누른 경우
        circle_center = (x, y)
        drawing = True
"""
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동 중
        if drawing:
            radius = max(abs(x - circle_center[0]), abs(y - circle_center[1]))

    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼을 뗀 경우
        drawing = False
        radius = max(abs(x - circle_center[0]), abs(y - circle_center[1]))
"""
# 카메라 열기
cap = cv2.VideoCapture(0, cv2.CAP_V4L)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', draw_circle)

while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.line(frame, (50, 50), (300, 300), (0, 255, 0), 5)
    cv2.rectangle(frame, (80, 80), (270, 270), (0, 0, 255), 5)
    
    if drawing:
        cv2.circle(frame, circle_center, radius, (255, 255, 0), 3, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, 'you', (80, 50), font, 4, (255, 255, 0), 5)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()