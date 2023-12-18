import numpy as np
import cv2

color = cv2.imread("data/strawberry.jpg", cv2.IMREAD_COLOR)
# color = cv2.imread("data/strawberry_dark.jpg", cv2.IMREAD_COLOR)
print(color.shape)

height, width, channels = color.shape
cv2.imshow("Original Image", color)
cv2.waitKey(0)

b, g, r = cv2.split(color)
bgr_split = np.concatenate((b, g, r), axis=1)
cv2.imshow("BGR Channels", bgr_split)
cv2.waitKey(0)

hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)
hsv_split = np.concatenate((h, s, v), axis=1)
cv2.imshow("split HSV", hsv_split)
cv2.waitKey(0)

# quiz 2-3
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
r, g, b = cv2.split(color)
rgb_split = np.concatenate((r, g, b), axis=1)
cv2.imshow("RGB Channels", rgb_split)
cv2.waitKey(0)

# quiz 2-4
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray scale", gray)
cv2.waitKey(0)

cv2.destroyAllWindows()
