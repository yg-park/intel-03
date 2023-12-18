import numpy as np
import cv2
# 이미지 파일을 Read
color = cv2.imread("strawberry.jpg",cv2.IMREAD_COLOR)
print(color.shape)

height,width,channels=color.shape
cv2.imshow("Original Image",color)


b,g,r=cv2.split(color)
bgr_split=np.concatenate((b,g,r),axis=1)
cv2.imshow("BGR Image",bgr_split)


hsv=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)
hsv_split=np.concatenate((h,s,v),axis=1)
cv2.imshow("HSV Image",hsv_split)



rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
r,g,b=cv2.split(rgb)
rgb_split=np.concatenate((r,g,b),axis=1)
cv2.imshow("RGB Image",rgb_split)


gray=cv2.cvtColor(color,cv2.COLOR_RGB2GRAY)
cv2.imshow("gray Image",gray)


# 별도 x키 입력이 있을 때 까지 대기
cv2.waitKey()




