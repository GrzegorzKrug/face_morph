import imutils
import random
import numpy as np
import time
import cv2
import os

from PIL import ImageFont, ImageDraw, Image
from io import BytesIO

image = cv2.imread("avatars/Youshisu.png")
eject_picture = cv2.imread("imposter.png")
rows, cols, channel = image.shape

"""
Script to rework offline functions
"""


def _create_imposter_image(image, name):
    imp_path = os.path.join(os.path.dirname(__file__), "src_images", "imposter.png")
    eject_picture = cv2.imread(imp_path)
    is_imposter = random.choice([True, False])
    angle = np.random.randint(15, 170)

    if is_imposter:
        text = "was the imposter."
    else:
        text = "was not the imposter."

    imposter = cv2.resize(image, (320, 320))
    imposter = imutils.rotate_bound(imposter, angle)

    rs, cs, _ = imposter.shape
    posx, posy = 350, 110
    roi = eject_picture[posy:posy + rs, posx:posx + cs, :]
    roi[:, :, :] = imposter

    text_color = (0, 0, 255) if is_imposter else (255, 255, 255)
    cv2.putText(eject_picture, name, (50, 600), cv2.FONT_HERSHEY_COMPLEX, 1.6, text_color, 2, cv2.LINE_AA)
    cv2.putText(eject_picture, text, (50, 660), cv2.FONT_HERSHEY_COMPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    dest_x, dest_y = eject_picture.shape[0:2]

    dest_x = int(dest_x // 1.5)
    dest_y = int(dest_y // 1.5)

    eject_picture = cv2.resize(eject_picture, (dest_y, dest_x))
    return eject_picture


def convert_to_sephia(image, depth=8, intense=40):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image, dtype=int)

    image[:, :, 0] = np.mean([gray, image[:, :, 0] / 3 - intense], axis=0)
    image[:, :, 1] = np.mean([gray, image[:, :, 1] / 3 + intense], axis=0)
    image[:, :, 2] = np.mean([gray, image[:, :, 2] / 3 + (depth ** 2) + intense], axis=0)

    mask = image > 255
    image[mask] = 255
    mask = image < 0
    image[mask] = 0

    image = np.array(image, dtype=np.uint8)
    return image


def _create_wanted_image(image, name, reward: int):
    img_path = os.path.join(os.path.dirname(__file__), "src_images", "wanted.jpg")
    wanted = cv2.imread(img_path)
    wanted = imutils.resize(wanted, width=310)

    crop_x = 1
    crop_y = 30
    cx, cy = 29, 90
    image = image[crop_y:-crop_y, crop_x:-crop_x, :]
    image = convert_to_sephia(image, 8, 40)
    rows, cols, channels = image.shape

    slic = (slice(cy, cy + rows), slice(cx, cx + cols), slice(None))
    roi = wanted[slic]
    roi[:, :] = image
    font_path = os.path.join(os.path.dirname(__file__), "src_fonts/BouWeste.ttf")

    wanted = image_array_to_pillow(wanted)
    draw = ImageDraw.Draw(wanted)

    font = get_font_to_bounding_box(font_path, name, max_width=257, max_height=None, initial_font_size=48)
    draw.text((155, 335), name, font=font, fill=(55, 40, 30), anchor="mm")

    reward = "$ " + f"{reward:,}"
    font = get_font_to_bounding_box(font_path, reward, max_width=250, max_height=None, initial_font_size=48)
    draw.text((280, 377), reward, font=font, fill=(30, 90, 20), anchor="rm")

    font = ImageFont.truetype(font_path, size=16)
    draw.text((35, 400), "For being in wrong place\n at wrong time.", font=font, fill=(50, 50, 50))

    wanted = image_pillow_to_array(wanted)
    return wanted


def get_font_to_bounding_box(font_path, text, max_width, max_height, initial_font_size=50, minimal_font_size=3):
    if max_width:
        text_width = max_width + 1
    else:
        text_width = None
    if max_height:
        text_height = max_height
    else:
        text_height = None

    font_size = initial_font_size + 1
    font = None

    while (
            (max_width and text_width > max_width) or
            (max_height and text_height > max_height)) \
            and font_size > minimal_font_size:
        font_size -= 1
        font = ImageFont.truetype(font_path, size=font_size)
        text_width, text_height = font.getsize(text)

    return font


def image_array_to_pillow(matrix):
    success, img_bytes = cv2.imencode(".png", matrix)
    bytes_like = BytesIO(img_bytes)
    image = Image.open(bytes_like)
    return image


def image_pillow_to_array(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    matrix = np.frombuffer(buffer.read(), dtype=np.uint8)
    matrix = cv2.imdecode(matrix, cv2.IMREAD_COLOR)
    return matrix


image = _create_wanted_image(image, "Youshisu", 50000000_000)

cv2.imshow("Image", image)
cv2.waitKey(15_000)
cv2.destroyAllWindows()
