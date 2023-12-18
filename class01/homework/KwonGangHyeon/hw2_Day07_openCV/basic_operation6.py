import numpy as np
import cv2


def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:    
        cv2.circle(param, (x, y), 50, (255, 0, 0), 2)
        cv2.imshow("draw", frame)

        print("여기 우선 들어옴..")
        cv2.waitKey(100)
        cv2.destroyWindow("draw")



cv2.namedWindow("Camera")

cap = cv2.VideoCapture(0)

topLeft = (50, 50)
bottomRight = (300, 300)



while (cap.isOpened()):
    ret, frame = cap.read()
    
    cv2.setMouseCallback("Camera", mouse_event, frame)

    # Line
    cv2.line(frame, topLeft, bottomRight, (0, 255, 0), 5)

    # Rectangle
    cv2.rectangle(frame, 
                  [pt + 30 for pt in topLeft], [pt - 30 for pt in bottomRight], 
                  (0, 0, 255), 5)
    
    # Circle
    cv2.circle(frame, 
               (topLeft[0] + 170, topLeft[1] + 170), 50,
               (255, 0, 0), 5)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'me',
                [pt+80 for pt in topLeft], font, 2, (0, 255, 255), 10)
    
    # Display
    cv2.imshow("Camera", frame)
    

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
