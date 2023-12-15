import numpy as np
import cv2

# 이미지 파일을 Read 하고 Color space 정보 출력
color = cv2.imread("strawberry.jpg", cv2.IMREAD_COLOR)
#color = cv2.imread("strawberry_dark.jpg", cv2.IMREAD_COLOR)
print(color.shape)

height,width,channels = color.shape
cv2.imshow("Original Image", color)

# Color channel 을 R, G, R로 분할하여 출력
b, g, r = cv2.split(color)
rgb_split = np.concatenate((b,g,r), axis=1)
cv2.imshow("BGR Channels", rgb_split)

# 색공간을 BGR 에서 HSV로 변환
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

# Channel을 H,S,V 로 분할하여 출력
h,s,v = cv2.split(hsv)
hsv_split = np.concatenate((h,s,v), axis=1)
cv2.imshow("Split HSV", hsv_split)

# HSV에서 RGB로 변환
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
cv2.imshow("HSV to RGB", rgb)

# RGB를 흑백(그레이스케일) 이미지로 변환
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
cv2.imshow("Grayscale", gray)

cv2.waitKey(0)  # 사용자 입력을 무한히 기다림
cv2.destroyAllWindows()  # 창 닫기
