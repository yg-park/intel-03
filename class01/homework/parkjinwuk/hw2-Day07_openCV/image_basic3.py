import numpy as np
import cv2

# 이미지 파일을 Read
img = cv2.imread("my_input.jpg")

# Crop 300x400 from original image from (100, 50)=(x,y)
cropped = img[50:450, 100:400]

# Resize cropped image from 300x400 to 400x200
resized = cv2.resize(cropped, (400,200))
bigsize = cv2.resize(img, (0,0), fx=1.5 , fy=1.5, interpolation = cv2.INTER_AREA)

# Display all
cv2.imshow("Original", img)
cv2.imshow("Cropped image", cropped)
cv2.imshow("Resized image", resized)
cv2.imshow("bigsize image", bigsize)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 1.Input image를 본인이 좋아하는 인물 사진으로 변경해서 적용하자. 그리고 본인이 사용한 input image의 size를 확인해 보자

# 2.본인이 사용한 이미지의 얼굴 영역만 crop해서 display해보자.
# 3.원본 이미지의 정확히 1.5배만큼 이미지를 확대해서 파일로 저장해보자.
# 4.openCV의 rotate API를 사용해서 우측으로 90도만큼 회전된 이미지를 출력해보자.
