import numpy as np
import cv2
import sys

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)

w = 176#640#352#320#176#160
h = 144#480#288#240#144#120
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

# videoFileName = 'data/output.mp4'
# # w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # width
# # h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height
# fps = cap.get(cv2.CAP_PROP_FPS) #frame per second
# fourcc = cv2.VideoWriter_fourcc(*'DIVX') #fourcc
# delay = round(1000/fps) #set interval between frame

# #Save Video
# out = cv2.VideoWriter(videoFileName, fourcc, fps, (w,h))
# if not (out.isOpened()):
# 	print("File isn't opend!!")
# 	cap.release()
# 	sys.exit()


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting...")
        break

    cv2.imshow("Camera", frame)
    # out.write(frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()

