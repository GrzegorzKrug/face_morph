import numpy as np
import cv2
import os

video = cv2.VideoCapture(0)

ORANGE = False
DARK = False

while True:
    _, original = video.read()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    h, w, c = original.shape

    laplacian = cv2.Laplacian(original, cv2.CV_64F)
    sobelx = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=5)
    output_combo = original

    edges = cv2.Canny(gray, 120, 150)

    cv2.imshow("Me", original)
    cv2.imshow("laplacian", laplacian)
    # cv2.imshow("sobelx", sobelx)
    # cv2.imshow("sobely", sobely)
    cv2.imshow("canny", edges)

    key = cv2.waitKey(delay=10)
    if key == ord("q"):
        break
