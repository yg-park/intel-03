import cv2
import numpy as np

img = cv2.imread("karina.jpg")

print(img.shape)
h = int(img.shape[0] * 1.5)
w = int(img.shape[1] * 1.5)
print(h, w)

cropped = img[20:450, 200:500]


resized = cv2.resize(img, (h, w))
cv2.imwrite("big_karina.png", resized)

rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("Original", img)
cv2.imshow("Cropped Image", cropped)
cv2.imshow("Resized Image", resized)
cv2.imshow("Rotated Image", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()