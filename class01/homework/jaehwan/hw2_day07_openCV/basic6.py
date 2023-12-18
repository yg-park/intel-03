import cv2

#read from the first camera device
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

topLeft=(50,50)
bottomRight=(300,300)

#성공적으로 video device 가 열렸으면 while 문 반복
while(cap.isOpened()):
    #한 프레임을 읽어옴
    ret,frame=cap.read()
    #Line
    cv2.line(frame,topLeft,bottomRight,(0,255,0),5)
    #rectangle
    cv2.rectangle(frame,[pt+30 for pt in topLeft],[pt-30 for pt in bottomRight],(0,0,255),5)
    #text
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'me',[pt+80 for pt in topLeft],font,2,(0,255,255),10)
    #display
    cv2.imshow("camera",frame)
