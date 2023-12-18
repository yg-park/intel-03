# Project ABC

* OpenCV 연습

## image_basic01.py

* 이미지 불러오기, 출력, 저장

## image_basic02.py

* 이미지 색 공간 표현

## image_basic03.py

* 이미지 자르기, 사이즈 변경, 회전

## image_basic04.py

* 동영상 불러오기, 출력, 사이즈 변경, 무한 반복, 캡쳐

## image_basic05.py

* 카메라 불러오기, 저장

## image_basic06.py

* 카메라 영상에 그림 그리기, 글씨 쓰기, 마우스 이벤트

```shell
def Mouse(event,x,y,flag,param) :
    global frame
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(frame, (x,y), 60, (255,0,0), 9)
```
```shell
 cv2.putText(텍스트를 추가할 이미지, 텍스트 내용, 텍스트 왼쪽 아래 모서리 좌표, 글꼴, 텍스트 크기, 텍스트 색상, 굵기(선택), 선타입(선택))
```

## image_basic07.py

* OpenCV GUI trackbar

```shell
cv2.createTrackbar(트랙바 이름, 트랙바를 붙일 창 이름, 초기값, 최대값, 값 변경 함수)
```
