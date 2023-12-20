import cv2

topLeft=(50,50)
bold=0
#callback function for the trackbar
def on_bold_trackbar(value):
    #print("trackbar value:", value)
    global bold
    bold = value

cv2.namedWindow("Camera")
cv2.createTrackbar("bold","Camera",bold,10,on_bold_trackbar)

#성공적으로 video device가 열렸으면 while 문 반복
while(cap.isOpened()):
    #한 프레임을 읽어옴
    ret,frame=cap.read()
    if ret is False:
        print("cant receive frame (stream end) exiting...")
        break

    #text
    cv2.puttext(frame,"TEXT",topLeft,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),1+bold)
    #display
    cv2.imshow("Carema",frame)

