import cv2

img = cv2.imread('my_input2.jpg')

cropped = img[50:450, 100:400]

resized = cv2.resize(cropped, (400, 200))

width, height, _ = img.shape
enlarged = cv2.resize(img, (int(width*1.5), int(height*1.5)))

rotate_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('Original', img)
cv2.imshow('Cropped image', cropped)
cv2.imshow('Resized image', resized)
cv2.imshow('enlarged image', enlarged)
cv2.imshow('rotate', rotate_img)

key = cv2.waitKey(0)

cv2.imwrite('enlarged.jpg', enlarged)

cv2.destroyAllWindows()
