import numpy as np
import cv2

img = cv2.imread("panda.jpg")

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

print(img.shape)

cv2.imshow("image", img)

cv2.waitKey(0)

# 저장
cv2.imwrite("output.png", img)
cv2.destroyAllWindow()
