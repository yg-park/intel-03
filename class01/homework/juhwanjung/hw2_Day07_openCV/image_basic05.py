import numpy as np
import cv2

cap = cv2.VideoCapture(0)

w = 650
h = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

fps = 24
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

while(cap.isOpened()) :
    ret, frame = cap.read()

    if ret is False :
        print("Can't receive frame (stream end?). Exiting ...")
        break

    
    

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q') :
        
        break
    
    out.write(frame)
    cv2.imshow("Camera", frame)

out.release()
cap.release()
cv2.destroyAllWindows()