import numpy as np
import cv2
import os


def start_video():
    vid = cv2.VideoCapture(0)
    return vid


eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

gray = True
cont = False
lapl = False
blur = False
sobel = False
edges = False

cx = 150
cy = 80

accuracy = 0.85

vid = start_video()
template = cv2.imread("eye.png")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

while True:
    success, bgr_img = vid.read()

    if not success:
        vid = start_video()
        success, bgr_img = vid.read()

    # height, width, channels = bgr_img.shape

    gray_img = bgr_img.copy()

    if gray:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    if blur:
        gray_img = cv2.GaussianBlur(gray_img, (5, 5), 1)

    if lapl:
        gray_img = cv2.Laplacian(gray_img, cv2.CV_64F)

    if edges:
        gray_img = cv2.Canny(gray_img, cy, cx)

    pos = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(pos >= accuracy)

    "Drawing Matches"
    for pt1 in zip(*loc):
        size_offset = 10
        pt1 = (pt1[1] - size_offset, pt1[0] - size_offset)
        height, width = template.shape
        pt2 = (pt1[0] + height + size_offset * 2, pt1[1] + width + size_offset * 2)
        cv2.rectangle(bgr_img, pt1, pt2, (0, 255, 0), 2)

    faces = face_cascade.detectMultiScale(gray_img, 1.5, 8)
    for (x, y, w, h) in faces:
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(bgr_img, pt1, pt2, (0, 255, 255), 2)
    eyes = eye_cascade.detectMultiScale(gray_img, 1.3, 8)
    for (x, y, w, h) in eyes:
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(bgr_img, pt1, pt2, (0, 255, 155), 2)

    cv2.imshow("Frame", bgr_img)

    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        gray = gray ^ True
        print(f"Gray is now: {gray}")
    elif key == ord('c'):
        cont = cont ^ True
    elif key == ord('l'):
        lapl = lapl ^ True
    elif key == ord('b'):
        blur = blur ^ True
    elif key == ord('s'):
        sobel = sobel ^ True
    elif key == ord('e'):
        edges = edges ^ True
    elif key == ord('p'):
        progress = progress ^ True
