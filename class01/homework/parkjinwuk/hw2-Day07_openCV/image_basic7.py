import numpy as np
import cv2

# Read from the first camera device
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

topLeft = (50, 50)
bold = 0
# Callback function for the trackbar
def on_bold_trackbar(value):
	#print("Trackbar value:", value)
	global bold
	bold = value

cv2.namedWindow("Camera")
cv2.createTrackbar("bold", "Camera", bold, 10, on_bold_trackbar)

# 성공적으로 video device가 열렸으면 while문 반복
while(cap.isOpened()):
	# 한 프레임을 읽어옴
	ret, frame = cap.read()
	if ret is False:
		print("Can't receive frame (stream end?). Exiting ...")
		break
	
