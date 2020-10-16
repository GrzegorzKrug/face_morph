import numpy as np
import glob
import cv2
import os

avatars = glob.glob("avatars_orig/*png")
avatars.sort()
export_dir = "avatars_ready"

os.makedirs(export_dir, exist_ok=True)

MAX_SIZE = 256

for avatar in avatars:
    print(avatar)
    bgr_img = cv2.imread(avatar)
    width, height, channels = bgr_img.shape

    if width > MAX_SIZE or height > MAX_SIZE:
        width = MAX_SIZE if width > MAX_SIZE else width
        height = MAX_SIZE if height > MAX_SIZE else height

        dim = (width, height)
        bgr_img = cv2.resize(bgr_img, dim)
    new_img_path = os.path.join(export_dir, os.path.basename(avatar))

    cv2.imwrite(new_img_path, bgr_img)
