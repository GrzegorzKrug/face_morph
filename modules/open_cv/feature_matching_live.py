import numpy as np
import imutils
import time
import cv2

pen_image = cv2.imread("src_images/pen2.png", cv2.IMREAD_COLOR)
# pen_image = imutils.resize(pen_image, height=800)
# pen_image = pen_image[100:-100, 100:-100, :]
# pen_image = np.array(pen_image, dtype=np.uint8)

print(pen_image.shape)
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
    template = cv2.cvtColor(pen_image, cv2.COLOR_BGR2GRAY)
    target = gray
    # template = pen_image

    "Key points, descriptors"
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(target, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    try:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        img3 = cv2.drawMatches(template, kp1, target, kp2, matches[:30], None, flags=2)
        cv2.imshow("Image", img3)
    except Exception:
        pass

    key = cv2.waitKey(100)
    if key == ord("q"):
        break
    elif key == 32:
        cv2.imwrite("capture.png", frame)
        print("Frame captured")
    elif key > 0:
        print("Pressed:", key)
