import numpy as np
import cv2

# 이미지 파일을 Read
img = cv2.imread("my_input.jpg")

# Image 란 이름의 Display 창 생성
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# Numpy ndarray Height/Width/Channel order
print(img.shape)

# Read 한 이미지 파일을 Display
cv2.imshow("image", img)

# 저장 방법
while True:
    key = cv2.waitKey(0)
    if key == ord('s'):
        # 's' 키를 누르면 이미지를 저장한 후 종료
        cv2.imwrite('saved_image.jpg', img)
        break
    else:
        # 's' 이외의 다른 키를 누르면 저장하지 않고 종료
        break

# output.png로 읽은 이미지 파일을 저장
#cv2.imwrite("output.png", img)

# Destroy all windows
cv2.destroyAllWindows()