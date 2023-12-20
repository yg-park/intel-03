import numpy as np
import cv2

def Mouse(event,x,y,flag,param) :
    global frame
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(frame, (x,y), 60, (255,0,0), 9)



cap = cv2.VideoCapture(0)

topLeft = (50,50)
bottomRight = (300, 300)

while(cap.isOpened()) :
    ret, frame = cap.read()

    cv2.line(frame, topLeft, bottomRight, (0,255,0), 5)

    cv2.rectangle(frame, [pt+30 for pt in topLeft], [pt-30 for pt in bottomRight], (0,0,255), 5)
    
    cv2.circle(frame, (300,300), 100, (255,0,0), 3)


    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'HUNGRY', [pt+8 for pt in topLeft], font, 2, (0,255,255), 10)
    
    

    key = cv2.waitKey(1)
    cv2.namedWindow('Camera')
    cv2.setMouseCallback("Camera", Mouse, frame)

    
    cv2.imshow("Camera", frame)

    if key & 0xFF == ord('q') :
        break