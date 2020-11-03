import numpy as np
import imutils
import cv2


def start():
    cap = cv2.VideoCapture(0)
    return cap


def draw_bbox(frame, focus):
    pt1, pt2 = focus
    cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)


video = start()

focus = ((200, 200), (500, 500))

_, frame = video.read()

gc_mask = np.zeros(frame.shape[:2], np.uint8)
back_model = np.zeros((1, 65), np.float64)
fore_model = np.zeros((1, 65), np.float64)

while True:
    is_next_frame, frame = video.read()

    if not is_next_frame:
        print("Start again...")
        video = start()
        _, frame = video.read()

    "Grab cut | foreground extraction"
    pt1, pt2 = focus
    rect = *pt1, *pt2

    cv2.grabCut(frame, gc_mask, rect, back_model, fore_model, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((gc_mask == 0) | (gc_mask == 2), 0, 1).astype("uint8")
    frame = frame * mask[:, :, np.newaxis]

    draw_bbox(frame, focus)

    "Display"
    cv2.imshow("Frame", frame)

    "User input"
    key = cv2.waitKey(508) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        video, tracker = start()
        print(tracker)

    elif key == 81:
        pos = video.get(cv2.CAP_PROP_POS_MSEC)
        video.set(cv2.CAP_PROP_POS_MSEC, pos - 5_000)
        print("Going back")
    elif key == 83:

        pos = video.get(cv2.CAP_PROP_POS_MSEC)
        video.set(cv2.CAP_PROP_POS_MSEC, pos + 5_000)
        print("Going forward")
        # forward
