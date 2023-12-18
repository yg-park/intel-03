import cv2

cap = cv2.VideoCapture("ronaldinho.mp4")

count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        # 비디오의 끝에 도달하면 비디오 포지션을 처음으로 되돌림
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 프레임의 크기가 비어있지 않은 경우에만 크기 조절
    if frame is not None:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(33)
        if key & 0xFF == ord('c'):
            cv2.imwrite(f'{count}.jpg', frame)
            count += 1
        if key & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
