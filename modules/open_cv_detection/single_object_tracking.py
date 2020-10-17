import numpy as np
from scipy import ndimage
import time
import cv2
import os


def start():
    cap = cv2.VideoCapture("people-walking.mp4")
    tracker = []
    return cap, tracker


video, tracker = start()
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100)

"First frame is always changed"
ret, frame = video.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
fgmask = fgbg.apply(frame_gray)

os.makedirs("frames", exist_ok=True)
RADIUS = 50
FRAME_SMOOTH = 15
tracking_point = (350, 180)

while True:
    ret, frame = video.read()

    if not ret:
        # break
        video, tracker = start()
        ret, frame = video.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    highlighted = frame.copy()
    height, width, _ = frame.shape

    fgmask = fgbg.apply(frame_gray)[:, :, np.newaxis]
    # good_mask = cv2.blur(fgmask, (3, 3))
    good_mask = cv2.erode(fgmask, (3, 3), iterations=3)
    # # new_mask = cv2.dilate(new_mask, (3, 3), iterations=10)
    # good_mask = cv2.blur(good_mask, (3, 3))
    good_mask = good_mask[:, :, np.newaxis]

    if tracker:
        last_pos = tracker[-1]
        x_, y_ = last_pos
        xmin = x_ - RADIUS * 2
        xmax = x_ + RADIUS * 2
        ymin = y_ - RADIUS * 2
        ymax = y_ + RADIUS * 2

        if xmin < 0:
            xmin = 0
        if xmax > width:
            xmax = width
        if ymin < 0:
            ymin = 0
        if ymax > height:
            ymax = height

        slice_of_interest = slice(ymin, ymax), slice(xmin, xmax), slice(None)
        temp = good_mask[slice_of_interest].copy()
        good_mask[:, :, :] = 0
        good_mask[slice_of_interest] = temp

    red_mask = (0, 0, 1) * good_mask
    # print(red_mask.shape)
    red_mask = np.where(red_mask > 5)
    highlighted[red_mask] = 250

    mass_y, mass_x, mass_z = ndimage.center_of_mass(good_mask)

    try:
        mass_center = (int(mass_x), int(mass_y))
    except ValueError:
        mass_center = tracking_point

    if mass_center:
        if tracker is []:
            current_pos = mass_center
        else:
            current_pos = tracker[-FRAME_SMOOTH:]
            current_pos.append(mass_center)
            current_pos = np.mean(current_pos, axis=0, dtype=int)

        current_pos = tuple(current_pos)
        tracker.append(mass_center)

        cv2.circle(frame, current_pos, RADIUS, (0, 105, 255))
        cv2.circle(highlighted, current_pos, RADIUS, (0, 105, 255))

    combined = np.concatenate([frame, highlighted], axis=1)
    both_masks = np.concatenate([fgmask, good_mask], axis=1)

    cv2.imshow("Raw mask and processed mask", both_masks)
    cv2.imshow("Original and Highlighted", combined)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        video, tracker = start()
        print(tracker)

cv2.destroyAllWindows()
