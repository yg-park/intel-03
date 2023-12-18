import numpy as np
import cv2
import os

# Qt 애플리케이션 시작 전에 환경 변수 설정
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 이미지 파일을 read
img = cv2.imread("du.jpg")

# Image 란 이름의 Display 창 생성
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# Numpy ndarray H/W/C order 이미지의 높이, 너비, 채널 수를 출력함
print(img.shape)

# Read 한 이미지 파일을 Display
cv2.imshow("image", img)

# 별도 키 입력이 있을 때까지 대기
# s를 누르면 output.png로 읽은 이미지 파일을 저장
key = cv2.waitKey(0)
if key & 0xFF == ord('s'):
    cv2.imwrite("output.png", img)
else:
    cv2.destroyAllWindows()

# Destroy all windows
cv2.destroyAllWindows()


