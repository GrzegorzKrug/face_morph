import numpy as np
import time
import glob
import cv2
import os

all_pics_paths = glob.glob("choosen/*png")
print(len(all_pics_paths))

all_pics_paths.sort(key=lambda name: int(name[-7:-4]))

first = None
whole_pic = None
add_numbers = True

for number, pic_path in enumerate(all_pics_paths):
    num_text = f"{number + 1:>03}"
    amplitude = 180
    offset = 50

    red = int(offset + abs(amplitude * np.cos(number / 200)))
    green = int(offset + abs(amplitude * np.cos(-30 + number / 70)))
    blue = int(offset + abs(amplitude * np.sin(50 + number / 50)))

    if not number % 60:
        if first is not None:
            if whole_pic is not None:
                print(whole_pic.shape, first.shape)
                whole_pic = np.concatenate([whole_pic, first], axis=1)
            else:
                whole_pic = first

        first = cv2.imread(pic_path)
        dimy, dimx, channels = first.shape
        if add_numbers:
            cv2.putText(
                    first,
                    num_text,
                    (250, 25),  # Orig
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,  # font size)
                    (red, green, blue, 255),  # font color
                    2,  # font stroke
            )
        # continue
    else:
        pic = cv2.imread(pic_path)
        if add_numbers:
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

temp = np.zeros((whole_pic.shape[0], first.shape[1], 3))
# temp = first
sh1, sh2, sh3 = first.shape

temp[:sh1, :sh2, :sh3] = first
print(whole_pic.shape, temp.shape)
whole_pic = np.concatenate([whole_pic, temp], axis=1)

# cv2.imwrite("All_nicks.png", first)
cv2.imwrite("all_nicks_with_numbers.png", whole_pic)
