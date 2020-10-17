import numpy as np
import cv2


def open_video():
    cap = cv2.VideoCapture("people-walking.mp4")
    return cap


video = open_video()
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100)

while True:
    ret, frame = video.read()
    if not ret:
        video = open_video()
        ret, frame = video.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    highlighted = frame.copy()
    fgmask = fgbg.apply(frame_gray)[:, :, np.newaxis]

    new_mask = cv2.erode(fgmask, (3, 3), iterations=3)
    # new_mask = cv2.dilate(new_mask, (3, 3), iterations=10)
    new_mask = cv2.blur(new_mask, (3, 3))
    new_mask = new_mask[:, :, np.newaxis]

    red_mask = (0, 0, 1) * new_mask

    red_mask = np.where(red_mask > 5)

    highlighted[red_mask] = 250

    combined = np.concatenate([frame, highlighted], axis=1)
    both_masks = np.concatenate([fgmask, new_mask], axis=1)

    cv2.imshow("Raw mask and processed mask", both_masks)
    cv2.imshow("Original and Highlighted", combined)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        video = open_video()

cv2.destroyAllWindows()
