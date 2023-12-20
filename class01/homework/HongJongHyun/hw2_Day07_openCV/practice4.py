import numpy as np
import cv2

cap = cv2.VideoCapture("ronaldinho.mp4")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
print(width) # 640
print(height) # 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320) # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width/2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200) # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height/2)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
print(width) # 640
print(height) # 480

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret is False:
		print("Can't receive frame. Exit")
		break

	cv2.imshow("Frame", frame)

	key = cv2.waitKey(50)
	if key & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
