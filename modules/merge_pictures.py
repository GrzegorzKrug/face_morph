import numpy as np
import time
import glob
import cv2
import os

all_pics_paths = glob.glob("choosen/*png")
print(len(all_pics_paths))

all_pics_paths.sort(key=lambda name: int(name[-7:-4]))

first = cv2.imread(all_pics_paths[0])
dimy, dimx, channels = first.shape

for number, pic_path in enumerate(all_pics_paths):
    num_text = f"{number:>03}"

    pic = cv2.imread(pic_path)
    red = int(abs(255 * np.cos(number / 700)))
    green = int(abs(255 * np.cos(70 + number / 150)))
    blue = int(abs(255 * np.sin(150 + number / 50)))

    if number == 1:
        cv2.putText(
                first,
                "001",
                (250, 25),  # Orig
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,  # font size)
                (red, green, blue, 255),  # font color
                2,  # font stroke
        )
        continue
    cv2.putText(
            pic,
            num_text,
            (250, 25),  # Orig
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,  # font size)
            (red, green, blue, 255),  # font color
            2,  # font stroke
            cv2.LINE_AA,
    )
    first = np.concatenate([first, pic], axis=0)

cv2.imwrite("All_nicks.png", first)
