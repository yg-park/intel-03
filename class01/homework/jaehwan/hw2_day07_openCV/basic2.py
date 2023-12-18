"""
1.각 색공각의 표현 방법 이해
2.HSV가 어떤 경우에 효과적? 색을 직관적으로 표현 가능
3.HSV를 BGR가 아닌 RGB로 다시 변환해서 출력
4.COLOR_RBG2GRAY를 사용해서 흑백으로 변환해 출력
"""
import cv2
import numpy as np
#이미지 파일을 Read 하고 Color space 정보 출력
color = cv2.imread("strawberry.jpg",cv2.IMREAD_COLOR)
color2 = cv2.imread("strawberry_dark.jpg",cv2.IMREAD_COLOR)
print(color.shape)

height,width,channels=color.shape
cv2.imshow("Original Image",color)

#Color channel 을 b,g,r로 분할하여 출력
b,g,r=cv2.split(color)
rgb_split=np.concatenate((b,g,r),axis=1)
cv2.imshow("BGR channels",rgb_split)
cv2.waitKey(0)

#색공간을 BGR에서 HSV로 변환
hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)

#channel 을 H,S,V로 분할하여 출력
h,s,v=cv2.split(hsv)
hsv_split=np.concatenate((h,s,v),axis=1)
cv2.imshow("Split HSV:",hsv_split)
cv2.waitKey(0)


