import numpy as np
import cv2

cap = cv2.VideoCapture("data/ronaldinho.mp4")

num = 1

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret is False:
        print("one cycle end...")
        cap.release()
        cap = cv2.VideoCapture("data/ronaldinho.mp4")
        continue

    frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(33) 
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('c'):
        file_name = str(num).rjust(3, '0')
        cv2.imwrite(f"data/{file_name}.png", frame)
        num += 1


cap.release()
cv2.destroyAllWindows()
