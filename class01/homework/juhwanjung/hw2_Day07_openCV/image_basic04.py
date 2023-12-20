import numpy as np
import cv2

cap = cv2.VideoCapture("ronaldinho.mp4")

print(cap.get(cv2.CAP_PROP_FPS))
num = 1
while(cap.isOpened()) :
    ret, frame = cap.read()

    if ret is False :
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    resized = cv2.resize(frame, (int(h / 2), int(w / 2)))
    cv2.imshow("Frame", frame)
    cv2.imshow("Resize", resized)
    key =cv2.waitKey(33)
    if key & 0xFF == ord('q') :
        break
    elif key == ord('c') :
        cv2.imwrite(f"{num:0=3}.jpg", frame)
        num += 1

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT) :
        cap.open('ronaldinho.mp4')

cap.release()
cv2.destroyAllWindows()