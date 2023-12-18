"""
1. input image를 인물 사진으로 변경해서 적용하고 size 확인
2. 이미지의 얼굴 영역만 crop해서 display
3. 원본이미지의 1.5배만큼 이미지 확대해서 저장
3. opencv의 rotate API를 사용해서 우측으로 90도만큼 회전된 이미지 출력
"""
import cv2
import numpy as np
#이미지 파일을 read
img = cv2.imread("my_input.jpg")
h,w,c=img.shape
print("height",h)
print("width",w)
print("channel",c)
#Crop 300x400 from original image from (100,50)=(x,y)
cropped=img[50:450,100:400]

#resize cropped image from 300x400 to 400x200
resized=cv2.resize(cropped,(400,200))
resized2=cv2.resize(img,(int(h*1.5),int(w*1.5)))
img_rotate=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
#display all
cv2.imshow("original",img)
cv2.imshow("cropped image",cropped)
cv2.imshow("resized image",resized)
cv2.imshow("1.5 image",resized2)
cv2.imshow("rotated image",img_rotate)
cv2.waitKey(0)
cv2.imwrite("1.5output.png",resized2)
cv2.destroyAllWindows()
