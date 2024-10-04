import cv2

cam = cv2.VideoCapture(0)
result, image = cam.read()
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(image.shape)

cam.release()

