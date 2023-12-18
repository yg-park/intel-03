import numpy as np
import cv2

img = cv2.imread("my_input.jpg")

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

print(img.shape)

cv2.imshow("image", img)



key = cv2.waitKey(0)

if key == ord('s') :
    cv2.imwrite("output.png", img)
else :
    pass

cv2.destroyAllWindows()