import numpy as np
import cv2
import os


def start_video():
    vid = cv2.VideoCapture(0)
    return vid


gray = True
cont = False
lapl = False
blur = False
sobel = False
edges = False

cx = 150
cy = 80

accuracy = 0.8

vid = start_video()
template = cv2.imread("eye.png")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

while True:
    success, bgr_img = vid.read()

    if not success:
        vid = start_video()
        success, bgr_img = vid.read()

    # height, width, channels = bgr_img.shape

    img = bgr_img.copy()

    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur:
        img = cv2.GaussianBlur(img, (5, 5), 1)

    if lapl:
        img = cv2.Laplacian(img, cv2.CV_64F)

    if edges:
        img = cv2.Canny(img, cy, cx)

    pos = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(pos >= accuracy)

    for pt1 in zip(*loc):
        size_offset = 10
        pt1 = (pt1[1] - size_offset, pt1[0] - size_offset)
        height, width = template.shape
        pt2 = (pt1[0] + height + size_offset * 2, pt1[1] + width + size_offset * 2)
        cv2.rectangle(bgr_img, pt1, pt2, (0, 255, 0), 2)
    # if sobel:
    #     img = cv2.Sobel(img, cv2.CV_64F)

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
