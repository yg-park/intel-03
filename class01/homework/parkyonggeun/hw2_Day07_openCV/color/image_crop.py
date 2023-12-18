import numpy as np
import cv2

# 이미지 파일을 Read
img = cv2.imread("my_input.jpg")

# Crop 300x400 from original image from (100, 50)=(x, y)
cropped = img[50:450, 100:400]

# Resize cropped image from 300x400 to 400x200
resized = cv2.resize(cropped, (400,200))
# 이미지의 높이, 너비 계산
height, width = img.shape[:2]

# 이미지를 1.5배로 확대
new_height = int(height * 1.5)
new_width = int(width * 1.5)
big_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Display all
#cv2.imshow("Original", img)
cv2.imshow("Original", big_image)
cv2.imshow("Cropped image", cropped)
#cv2.imshow("Resized image", resized)

# 90도 회전
rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("Rotated image", rotated_img)

cv2.waitKey(0)  # 사용자 입력을 무한히 기다림
cv2.destroyAllWindows()  # 창 닫기