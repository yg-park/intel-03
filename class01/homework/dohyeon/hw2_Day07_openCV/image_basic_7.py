import cv2

cap = cv2.VideoCapture(0)

topLeft = (50, 150)
bold_value = 0
font_size = 0
r = 0
g = 0
b = 0

# Trackbar callback function
def on_bold_trackbar(b_value):
    global bold_value
    bold_value = b_value

def on_font_size_trackbar(f_value):
    global font_size
    font_size = f_value

def r_trackbar(r_value):
    global r
    r = r_value
    if r > 255:
        r = 255
    elif r < 0:
        r = 0

def g_trackbar(g_value):
    global g
    g = g_value
    if g > 255:
        g = 255
    elif g < 0:
        g = 0

def b_trackbar(b_value):
    global b
    b = b_value
    if b > 255:
        b = 255
    elif b < 0:
        b = 0

# If the video device is successfully opened, repeat the loop
while cap.isOpened():
    # Read one frame
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Text
    cv2.putText(frame, "TEXT", topLeft, cv2.FONT_HERSHEY_SIMPLEX, 2 + font_size, (0+b, 255+g, 255+r), 1 + bold_value)
    # Display
    cv2.imshow("Camera", frame)

    # Create trackbar with a different name
    cv2.createTrackbar("bold_trackbar", "Camera", bold_value, 10, on_bold_trackbar)

    cv2.createTrackbar("font_size_trackbar", "Camera", font_size, 10, on_font_size_trackbar)

    cv2.createTrackbar("r_trackbar", "Camera", r, 255, r_trackbar)

    cv2.createTrackbar("g_trackbar", "Camera", g, 255, g_trackbar)

    cv2.createTrackbar("b_trackbar", "Camera", b, 255, b_trackbar)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

