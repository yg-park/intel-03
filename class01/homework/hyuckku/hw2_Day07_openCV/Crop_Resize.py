import numpy as np
import cv2
import os

# Qt 애플리케이션 시작 전에 환경 변수 설정
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 이미지 파일을 Read
img = cv2.imread("du.jpg")

# Crop 300x400 from original image from (100,50) = (x,y)
cropped = img[20:110, 260:350] #260:350, 20:110]

# Resize cropped image from 300x400 to 400x200
resized = cv2.resize(img, (900,900))

#display all
cv2.imshow("Original", img)
cv2.imshow("Cropped image", cropped)
cv2.imshow("Resized image", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
