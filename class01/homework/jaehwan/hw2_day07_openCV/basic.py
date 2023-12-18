"""
1.print(img.shape)의 출력 결과는 무슨 의미일까? 입력한 이미지 출력
2.본인이 좋아하는 사진을 wab에서 다운받아서 OpencvAPI를 사용해서 Display 및 사진으로 저장
3. 소문자 s 입력으로 종료
"""
import numpy as np
import cv2

#이미지 파일을 read
img = cv2.imread("my_input.jpg")

#image 란 이름의 display 창 생성
cv2.namedWindow("image",cv2.WINDOW_NORMAL)

#numpy ndarray H/W/C order
print(img.shape)

#read 한 이미지 파일을 display
cv2.imshow("image",img)

#별도 키 입력이 있을때 까지 대기
while True:
    if cv2.waitKey(1) == ord('s'):
        break

#output.png 로 읽은 이미지 파일을 저장
cv2.imwrite("output.png",img)

#destroy all windows
cv2.destroyAllWindows()
