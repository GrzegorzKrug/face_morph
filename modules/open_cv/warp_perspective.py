import numpy as np
import imutils
import time
import cv2

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
book = cv2.imread("src_images/photo_wbook.png", cv2.IMREAD_COLOR)

while True:
    ret, frame = video.read()
    height, width, ch = frame.shape
    # warped = cv2.warpPerspective(frame,)
    origin = np.array([
            [76, 465],
            [274, 395],
            [227, 120],
            [1, 159],

    ], dtype=np.float32)
    new_points = np.array([
            [0, 465],
            [274, 465],
            [274, 159],
            [0, 159],

    ], dtype=np.float32)

    transform = cv2.getPerspectiveTransform(origin, new_points)
    warped = cv2.warpPerspective(book, transform, (width, height))
    # warped = warped[200:350, 200:350, :]

    cv2.imshow("Frame", frame)
    cv2.imshow("book", book)
    cv2.imshow("warped", warped)

    key = cv2.waitKey(100)
    if key == ord("q"):
        break
    elif key == 32:
        cv2.imwrite("capture.png", frame)
        print("Frame captured")
    elif key > 0:
        print("Pressed:", key)
