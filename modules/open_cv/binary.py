import numpy as np
import imutils
import time
import cv2

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_BUFFERSIZE, 2)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    all_chans = []
    for chan in frame[:, :]:
        _, binary = cv2.threshold(chan, 70, 255, cv2.THRESH_BINARY)
        # binary = cv2.adaptiveThreshold(chan, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 1.9)
        all_chans.append(binary)

    all_chans = np.array(all_chans)

    "Mean color value and post conversion, not in mean!"
    chan_mean = all_chans.mean(axis=2).astype("uint8")

    kernel = np.ones((3, 3))
    kernel = kernel / kernel.sum()
    dil = cv2.dilate(chan_mean, kernel=kernel, iterations=1)
    ero = cv2.erode(chan_mean, kernel=kernel, iterations=2)

    # cv2.imshow("Frame", frame)
    # cv2.imshow("all_chans", all_chans)
    cv2.imshow("chan_mean", chan_mean)
    # cv2.imshow("dil", dil)
    # cv2.imshow("ero", ero)

    key = cv2.waitKey(100)
    if key == ord("q"):
        break
    elif key == 32:
        cv2.imwrite("capture.png", frame)
        print("Frame captured")
    elif key > 0:
        print("Pressed:", key)
