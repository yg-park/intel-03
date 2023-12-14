import numpy as np
import cv2

""" < 원본 실습 코드 >

img = cv2.imread("data/my_input.jpg")

cropped = img[50:450, 100:400] # 부분행렬 도출

resized = cv2.resize(cropped, (400, 200))

cv2.imshow("Original image", img)
cv2.imshow("cropped image", cropped)
cv2.imshow("resized cropped image", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""

# 퀴즈 적용 코드

img = cv2.imread("data/kimtaelee.jpg")
print(img.shape)

cropped = img[30:350, 210:450] # 김태리의 얼굴영역 부분행렬 도출
resized = cv2.resize(img, (int(img.shape[0] * 1.5), int(img.shape[1] * 1.5)))
img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 
 
cv2.imshow("Original image", img)
cv2.imshow("cropped image", cropped)
cv2.imshow("resized cropped image", resized)
cv2.imshow("90rotate image", img90)

cv2.waitKey(0)
cv2.destroyAllWindows()
