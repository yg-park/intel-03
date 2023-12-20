import numpy as np
import cv2
import os

# Qt 애플리케이션 시작 전에 환경 변수 설정
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 이미지 파일을 Read하고 Color space 정보 출력
#color = cv2.imread("strawberry.jpg", cv2.IMREAD_COLOR)
color = cv2.imread("strawberry_dark.jpg", cv2.IMREAD_COLOR)
print(color.shape)

height, width, channels = color.shape
cv2.imshow("Original Image",color)

#Color channel을 B,G,R로 분할하여 출력
b,g,r = cv2.split(color)
rgb_split = np.concatenate((b,g,r),axis=1)
cv2.imshow("BGR Channels",rgb_split)

# 색공간을 BGR에서 HSV로 변환
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

# Channel을 H,S,V로 분할하여 출력
h,s,v = cv2.split(hsv)
hsv_split = np.concatenate((h,s,v),axis=1)
cv2.imshow("Split HSV",hsv_split)

