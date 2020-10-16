import requests
import numpy as np
import cv2
import os

from hashlib import sha1

images_dir = "animals"
name_prefix = "tree"

os.makedirs(images_dir, exist_ok=True)

with open("trees.txt", "rt") as file:
    links = file.read().split("\n")

for link in links:
    hex = sha1(bytes(link, encoding='utf-8')).hexdigest()
    img_name = f"{name_prefix}-{hex}.png"
    img_path = os.path.join(images_dir, img_name)
    if os.path.isfile(img_path):
        print(f"This file exists already: {img_path}")
        continue

    try:
        re = requests.get(link, stream=True)

        if re.status_code != 200:
            print(f"Invalid, {re.status_code}, {link}")
            continue

    except Exception as err:
        print(f"Error: {err}")
        continue

    img_raw = re.content

    image = np.frombuffer(img_raw, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    try:

        cv2.imwrite(img_path, image)
        print(f"Saved image: {img_path}")
    except Exception as err:
        print(f"Error, {err}")

print(f"Finished")
