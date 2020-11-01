import numpy as np
import cv2
import os

video = cv2.VideoCapture(0)

while True:
    _, original = video.read()
    frame_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    h, w, c = original.shape
    lower_orange = np.array([0, 100, 128])
    upper_orange = np.array([8, 255, 255])

    mask_alpha = cv2.inRange(frame_hsv, lower_orange, upper_orange)

    mask_alpha = cv2.erode(mask_alpha, (3, 3), iterations=2)
    mask_alpha = cv2.dilate(mask_alpha, (5, 5), iterations=5)
    mask = mask_alpha > 0  # True if orange

    mask_pic = cv2.cvtColor(mask_alpha, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    pic1 = cv2.bitwise_and(gray, gray, mask=~mask_alpha)
    pic1 = cv2.cvtColor(pic1, cv2.COLOR_GRAY2BGR)
    pic2 = cv2.bitwise_and(original, original, mask=mask_alpha)

    output = pic1 + pic2
    # output = cv2.bitwise_or(pic1, pic2)

    out = np.concatenate([original, mask_pic, output], axis=1)
    cv2.imshow("Me", out)

    key = cv2.waitKey(delay=10)
    if key == ord("q"):
        break
