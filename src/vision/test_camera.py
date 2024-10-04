import cv2

cam = cv2.VideoCapture(0)
result, image = cam.read()
cv2.imshow("Image", image)
print(image.shape)

cam.release()

