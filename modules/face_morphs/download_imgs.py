import requests
import numpy as np
import cv2

with open("acustic.txt", "rt") as file:
    links = file.read().split("\n")

for link in links:
    re = requests.get(link, stream=True)
    if re.status_code != 200:
        print(f"Invalid, {re.status_code}")
        continue
    print(f"Downloading {link}")

    # re.raw.decode_content = True
    # img_raw = re.content
    img_raw = re.raw

    # img = np.frombuffer(img, np.uint8)
    # image = np.asarray(img_raw, dtype=np.uint8)
    image = np.asarray(bytearray(img_raw.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite("cat.png", image)
