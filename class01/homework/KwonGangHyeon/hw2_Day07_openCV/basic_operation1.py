import numpy as np
import cv2

# 이미지 파일을 read
img = cv2.imread("data/cheeseman.jpg")

# image란 이름의 display창 생성
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# Numpy ndarray H/W/C order
print(img.shape)

# read한 이미지 파일을 display
cv2.imshow("image", img)

# 별도 키 입력이 있을때까지 대기
key_input = cv2.waitKey(0)

# output.png로 읽은 이미지 파일을 저장
if key_input == ord('s'):
    cv2.imwrite("data/cheeseman-output.png", img)
else:
    pass

# Diestroy all windows
cv2.destroyAllWindows()

