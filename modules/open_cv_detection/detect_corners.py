import numpy as np
import cv2

image = cv2.imread("shapes.png")
image = cv2.imread("Testy.png")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

corners = cv2.goodFeaturesToTrack(image_gray, 1000, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 3, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey()
