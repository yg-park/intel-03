import numpy as np
import cv2
# 이미지 파일을 Read
img = cv2.imread("my_input.jpg")

w,h,ch=img.shape
cX=w*0.5
cY=h*0.5
cropped = img[130:355,50:395]


resized=cv2.resize(cropped,(400,200))
half_doubled=cv2.resize(img,(678,1016))


M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated_45 = cv2.warpAffine(img, M, (w, h))




cv2.imshow("Original", img)
cv2.imshow("cropped image", cropped)
cv2.imshow("resized image", resized)
cv2.imshow("half_doubled Image",half_doubled)
cv2.imshow("rotated_45 Image",rotated_45)


# output.png로 읽은 이미지 파일을 저장
cv2.imwrite("half_doubled.png", half_doubled)


# 별도 x키 입력이 있을 때 까지 대기
cv2.waitKey()
# Destroy all windows
cv2.destroyAllWindows()

