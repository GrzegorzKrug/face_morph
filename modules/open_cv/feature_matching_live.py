import numpy as np
import imutils
import time
import cv2

pen_image = cv2.imread("src_images/pen.jpg", 0)
pen_image = imutils.resize(pen_image, width=800)
pen_image = pen_image[100:-100, 100:-100]

orb = cv2.ORB_create()

# video = cv2.VideoCapture(0)

# def get_last_frame(video):
#     ret, frame = video.read()
#     while ret:
#         ret, frame = video.read()
#         print(ret)
#     return frame


video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_BUFFERSIZE, 2)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    "Key points, descriptors"
    kp1, des1 = orb.detectAndCompute(pen_image, None)
    kp2, des2 = orb.detectAndCompute(gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    try:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        img3 = cv2.drawMatches(pen_image, kp1, frame, kp2, matches[:50], None, flags=2)
        cv2.imshow("Image", img3)
    except Exception:
        pass

    key = cv2.waitKey(100)
    if key == ord("q"):
        break
    elif key == 32:
        cv2.imwrite("capture.png", img3)
    elif key > 0:
        print("Pressed:", key)
