import numpy as np
import cv2
import os

video = cv2.VideoCapture(0)

ORANGE = False
DARK = False

while True:
    _, original = video.read()
    blurded = cv2.medianBlur(original, 15)
    frame_hsv = cv2.cvtColor(blurded, cv2.COLOR_BGR2HSV)
    h, w, c = original.shape

    if ORANGE:
        lower = np.array([0, 100, 128])
        upper = np.array([8, 255, 255])
    elif DARK:
        lower = np.array([0, 0, 0])
        upper = np.array([255, 50, 100])
    else:
        lower = np.array([100, 20, 90])
        upper = np.array([123, 255, 255])

    # frame_hsv = cv2.blur(frame_hsv, (15, 15))
    initial_mask = cv2.inRange(frame_hsv, lower, upper)

    kernel = np.ones((5, 5))
    mask_alpha = cv2.erode(initial_mask, kernel, iterations=1)
    mask_alpha = cv2.dilate(mask_alpha, kernel, iterations=1)
    mask = mask_alpha > 0  # True if matching

    mask_pic = cv2.cvtColor(mask_alpha, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    pic1 = cv2.bitwise_and(gray, gray, mask=~mask_alpha)
    pic1 = cv2.cvtColor(pic1, cv2.COLOR_GRAY2BGR)
    pic2 = cv2.bitwise_and(original, original, mask=mask_alpha)

    output = pic1 + pic2
    # output = cv2.bitwise_or(pic1, pic2)
    fully_saturated = frame_hsv.copy()
    for x in range(50):
        fully_saturated[x, :256, 0] = np.arange(256)

    fully_saturated[:, :, 1] = 255
    fully_saturated[:, :, 2] = 255
    fully_saturated = cv2.cvtColor(fully_saturated, cv2.COLOR_HSV2BGR)
    initial_mask_bgr = cv2.cvtColor(initial_mask, cv2.COLOR_GRAY2BGR)
    out1 = np.concatenate([initial_mask_bgr, original, ], axis=1)
    out2 = np.concatenate([mask_pic, output, ], axis=1)

    out = np.concatenate([out1, out2], axis=0)

    "Morphing to get rid or negative false and positive false"
    openings = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel)
    closings = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel)
    morphs = np.concatenate([initial_mask, openings, closings], axis=1)
    cv2.imshow("initial mask / open / close", morphs)
    # cv2.MORPH_BLACKHAT
    # cv2.MORPH_TOPHAT

    cv2.imshow("Me", out)

    key = cv2.waitKey(delay=10)
    if key == ord("q"):
        break
