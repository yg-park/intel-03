import cv2
import numpy as np

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

color = cv2.imread("strawberry.jpg")

print(color.shape)

cv2.imshow("Original Image", color)

cv2.waitKey(0)

b, g, r = cv2.split(color)
rgb_split = np.concatenate((b,g,r), axis=1)
cv2.imshow("BGR Channels", rgb_split)

cv2.waitKey(0)

hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)
hsv_split = np.concatenate((h,s,v), axis=1)
cv2.imshow("Split HSV", hsv_split)

cv2.waitKey(0)

#rgb = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
