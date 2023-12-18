import numpy as np
import cv2

img = cv2.imread("minji.jpeg")

cropped = img[50:450, 100:400]
cropped2 = img[36:80, 100:170]
resized = cv2.resize(cropped2, (400,200))
cv2.imshow("Original", img)
cv2.imshow("cropped img", cropped2)
cv2.imshow("resized img", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
