import numpy as np
import scipy.misc as msc

import glob
import cv2
import os

all_pics_paths = glob.glob("choosen/*png")
all_pics_paths.sort(key=lambda name: int(name[-7:-4]))

first = cv2.imread(all_pics_paths[0])
dimy, dimx, channels = first.shape

print(f"X:{dimx}, Y:{dimy}")

ready = cv2.imread("All_nicks.png")
ready_gray = cv2.cvtColor(ready, cv2.COLOR_BGR2GRAY)

threshold = 0.9
os.makedirs("conflicts", exist_ok=True)

all_conflicts = ready.copy()
for number, pic_path in enumerate(all_pics_paths, 1):
    num_text = f"{number:>03}"

    template = cv2.imread(pic_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    results = cv2.matchTemplate(ready_gray, template, cv2.TM_CCOEFF_NORMED)
    locs = np.where(results >= threshold)

    if len(locs[0]) == 1:
        continue
    elif len(locs[0]) < 1:
        print(f"Not found: {num_text}")
        continue
    else:
        print(f"Conflict in {num_text}")

    new_image = ready.copy()
    color = np.random.randint(70, 255, 3).tolist()

    for pt in zip(*locs):
        posy, posx, = pt
        # print(x, y)
        pt1 = (posx, posy)
        pt2 = (posx + dimx, posy + dimy)

        print(f"{str(pt1):<10} -> {pt2}")

        cv2.rectangle(new_image, pt1, pt2, color, 10)
        cv2.rectangle(all_conflicts, pt1, pt2, color, 10)

    cv2.imwrite(f"conflicts/{num_text:>03}.png", new_image)
cv2.imwrite(f"conflicts/all.png", all_conflicts)
