import numpy as np
import cv2

#이미지 파일을 Read
img = cv2.imread("my_input.jpg")

#Crop 300x400 from original image from (100,50)=(x,y)
cropped = img[50:450, 100:400]

#Resize cropped image from 300x400 to 400x200
resized = cv2.resize(cropped, (400,200))

#Q3. 원본 이미지의 정확히 1.5배만큼 이미지를 확대해서 파일로 저장
dst = cv2.resize(img, None, fx=1.5, fy=1.5)

#Q4. openCV의 rotate API를 사용해서 우측으로 90도만큼 회전된 이미지를 출력
img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("rotate90", img90)
cv2.imshow("dst", dst)

cv2.imshow("Original", img)
cv2.imshow("Cropped image", cropped)
cv2.imshow("Resized image", resized)

cv2.waitKey(0)
cv2.destroyallWindows()


