import cv2

def draw_circle(event, x, y, flags, param):
    global circles
    if event == cv2.EVENT_LBUTTONDOWN:
        circles.append((x, y))

cap = cv2.VideoCapture(0)

circles = []

topLeft = (50, 50)
bottomRight = (300, 300)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    cv2.line(frame, topLeft, bottomRight, (0, 255, 0), 5)

    cv2.rectangle(frame,
                  (topLeft[0] + 30, topLeft[1] + 30),
                  (bottomRight[0] - 30, bottomRight[1] - 30), (0, 0, 255), 5)

    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame, 'dohyeon', (topLeft[0] + 100, topLeft[1] + 100),
                font, 1, (255, 255, 0), 10)

    for circle in circles:
        cv2.circle(frame, circle, 20, (0, 255, 0), 10)

    cv2.imshow("Camera", frame)

    cv2.setMouseCallback("Camera", draw_circle)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

