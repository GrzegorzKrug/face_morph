import imutils
import pickle
import random
import numpy as np
import copy
import glob
import time
import cv2
import os
import re

from itertools import repeat

AVATAR_SIZE = 50


def load_palette(config=None):
    if config:
        pal_path = config['palpath']
    else:
        pal_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "palette"))
        pal_path = os.path.join(pal_dir, "palettes.pickle")
    if os.path.isfile(pal_path):
        with open(pal_path, "rb") as fp:
            palette = pickle.load(fp)
    else:
        print(pal_path)
        raise FileNotFoundError("Palette does not exists")
    # print(f"Palette loaded, size: {len(palette)}")  # size 1
    return palette


def find_closes_image(arguments):
    (
            targ_blue, targ_green, targ_red,
            targ_hue, targ_light, targ_sat,
            palette
    ) = arguments
    error = 255 ** 2
    best = None
    for path, inner_dict in palette.items():
        red, green, blue = inner_dict['r'], inner_dict['g'], inner_dict['b']
        hue, light, sat = inner_dict['hue'], inner_dict['light'], inner_dict['sat']

        cur_error = abs(targ_blue - blue) \
                    + abs(targ_green - green) \
                    + abs(targ_red - red)
        # + abs(targ_light - light) * 0.5
        # + abs(targ_ - hue)

        if cur_error < error:

            error = cur_error
            best = path
    return best


def recreate_all_palettes():
    """
    Palettes are grouped by folder name, then by path:
    example:
    {"emotes":
        [
        {"path1": {"r":int, "g":int, "b":int}},
        {"path2": {"r":int, "g":int, "b":int}},
        ...
        ]
    }
    Returns:

    """
    pal_dirs = {}
    PAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "palette"))
    PAL_PATH = os.path.join(PAL_DIR, "palettes.pickle")

    for directory in os.listdir(PAL_DIR):
        cur_path = os.path.join(PAL_DIR, directory)
        if os.path.isdir(cur_path):
            arr = create_palette(cur_path)
            pal_dirs.update({directory: arr})

    with open(PAL_PATH, "wb") as fp:
        pickle.dump(pal_dirs, fp)


def make_stamp_square(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, c = image.shape
    if h < w:
        w = h
    elif w < h:
        h = w
    else:
        print(f"Images shape is the same, not editing: {img_path}")
        return None
    new_img = image[0:h, 0:w, :]
    cv2.imwrite(img_path, new_img)
    return new_img


def create_palette(dir_path):
    palette = {}
    all_images = glob.glob(f"{dir_path}/**/*.png", recursive=True)
    all_images += glob.glob(f"{dir_path}/**/*.jpg", recursive=True)

    for image_path in all_images:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        height, width, channels = image.shape
        image = imutils.resize(image, width=AVATAR_SIZE)

        if height < AVATAR_SIZE or width < AVATAR_SIZE:
            print(f"This image is too small: {height:>4}, {width:>4} - {image_path}")
            continue
        elif height != width:
            print(f"This image is not squared: {height:>4}, {width:>4} - {image_path}")
            image = make_stamp_square(image_path)
        hls_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        blue, green, red = np.mean(image, axis=0).mean(axis=0)
        hue, light, sat = np.mean(hls_img, axis=0).mean(axis=0)
        rel_dir = re.sub(r"^.*palette" + f"{os.path.sep}", "", image_path)
        # print(rel_dir)

        palette[rel_dir] = {
                "r": red, "g": green, "b": blue,
                "hue": hue, "sat": sat, "light": light,
        }
    print(f"Palette size: {len(palette.keys())}")
    return palette


def select_palette(all_palettes, selection=None):
    if selection:
        cur = [value for pal_name, value in all_palettes.items()
               for sel in selection if sel in pal_name]
        palette = {}
        for c in cur:
            palette.update({**c})

    else:
        palette = random.choice(list(all_palettes.values()))
    print(f"Palette size: {len(palette)}")
    return palette


outpath = "out"

PAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "palette"))
PAL_PATH = os.path.join(PAL_DIR, "palettes.pickle")

config = {
        "paldir": PAL_DIR, "palpath": PAL_PATH}

# recreate_all_palettes()
palette = load_palette(config)
palette = select_palette(palette, ["face"])

for image_name in os.listdir("src"):
    print(f"Now: {image_name}")
    name, _ = image_name.split(".")
    image_path = os.path.join("src", image_name)
    OUTPUT_GIF_MAX_SIZE = 800
    PROC_POWER = 50

    image = cv2.imread(image_path)
    height, width, ch = image.shape
    if height > width:
        image = imutils.resize(image, height=OUTPUT_GIF_MAX_SIZE)
    else:
        image = imutils.resize(image, width=OUTPUT_GIF_MAX_SIZE)

    height, width, ch = image.shape
    closest = image.copy()
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    PIXEL_RATIO = round(OUTPUT_GIF_MAX_SIZE / PROC_POWER)
    print(f"Pixel: {PIXEL_RATIO}")
    MAX_PIC_SIZE = OUTPUT_GIF_MAX_SIZE

    row_range = range(0, height, PIXEL_RATIO)
    col_range = range(0, width, PIXEL_RATIO)

    for r in row_range:
        for c in col_range:
            sl = slice(r, r + PIXEL_RATIO), slice(c, c + PIXEL_RATIO), slice(None, None)
            roi = image[sl]
            roi_hls = image_hls[sl]
            roi2 = closest[sl]

            avg = roi.mean(axis=0).mean(axis=0)
            hls = roi_hls.mean(axis=0).mean(axis=0)
            roi[:, :, :] = avg
            best_img = find_closes_image([*avg, *hls, palette])
            pal = os.path.join(PAL_DIR, best_img)
            best_img = cv2.imread(pal, cv2.IMREAD_COLOR)
            avg2 = best_img.mean(axis=0).mean(axis=0)
            roi2[:, :, :] = avg2

    if width * 1.5 < height and not (width > 1.5 * height):
        axis = 0
    else:
        axis = 1
    output = np.concatenate([image, closest], axis=axis)
    cv2.imwrite(f"{name}.png", output)
    cv2.imwrite(os.path.join("out", f"{name}.png"), image)
    cv2.imwrite(os.path.join("out", f"{name}-best.png"), closest)
