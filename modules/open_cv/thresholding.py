import numpy as np
import cv2
import os

image = cv2.imread("src_images/bookpage.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

retval, threshold = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 141, 1.8)
retval2, otsu = cv2.threshold(gray, 5, 255, cv2.THRESH_OTSU)

cv2.imshow("original", image)
cv2.imshow("threshold", threshold)
cv2.imshow("Adaptvie", gaus)
# cv2.imshow("Otsu", otsu)
cv2.waitKey(delay=15_000)
cv2.destroyAllWindows()
