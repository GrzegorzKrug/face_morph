from multiprocessing import Process, Queue, Pool

import imutils
import numpy as np
import glob
import time
import cv2
import sys
import os


def time_it_dec(func):
    def wrapper(*args, **kwargs):
        time0 = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        duration = time_end - time0
        if duration < 60:
            print(f"{func.__name__} was executed in {duration:>6.2f}s")
        else:
            print(f"{func.__name__} was executed in {duration/60:>6.2f}m")
        return result

    return wrapper


@time_it_dec
def create_palette():
    palette = {}
    all_images = glob.glob("square_stamps/**/*.png", recursive=True)
    all_images += glob.glob("square_stamps/**/*.jpg", recursive=True)

    for image_path in all_images:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        height, width, channels = image.shape
        image = imutils.resize(image, width=AVATAR_SIZE)
        if height < 15 or width < 15:
            print(f"This image is too small: {height:>4}, {width:>4} - {image_path}")
            continue
        elif height != width:
            print(f"This image is not squared: {height:>4}, {width:>4} - {image_path}")
            image = make_stamp_square(image_path)

        if USE_HSV:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, saturation, value = np.mean(hsv, axis=0).mean(axis=0)
        else:
            # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, saturation, value = np.mean(image, axis=0).mean(axis=0)
        palette[image_path] = {"h": hue, "sat": saturation, "val": value}
    print(f"Palette size: {len(palette.keys())}")
    return palette


# @time_it_dec
def find_closes_image(arguments):
    _palette, targ_h, targ_s, targ_v = arguments
    error = 255 ** 2
    best = None
    for path, inner_dict in _palette.items():
        h, s, v = inner_dict.values()
        cur_error = (targ_h - h) ** 2 + (targ_s - s) ** 2 + (targ_v - v) ** 2  # squared error
        if cur_error < error:

            error = cur_error
            best = path
    return best


@time_it_dec
def get_mozaic(targ_path, ignore_image_size=True, fill_border_at_error=False):
    target = cv2.imread(targ_path, cv2.IMREAD_COLOR)
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    h, w, c = target.shape

    palette = create_palette()
    if (h % PIXEL_RATIO or w % PIXEL_RATIO) and not ignore_image_size:
        print(f"Invalid size, H:{h}%->{h % PIXEL_RATIO}, W:{w}->{w % PIXEL_RATIO}")
        sys.exit(0)

    output = np.zeros(
            (h * AVATAR_SIZE // PIXEL_RATIO, w * AVATAR_SIZE // PIXEL_RATIO, 3),
            dtype=np.uint8
    )
    # que = Queue(20)
    pool = Pool(50)

    for row_num, cur_row in enumerate(range(0, h, PIXEL_RATIO)):

        time0 = time.time()
        "Make Queue"
        iter_like_orders = []
        for col_num, cur_col in enumerate(range(0, w, PIXEL_RATIO)):
            target_slice = slice(cur_row, cur_row + PIXEL_RATIO), slice(cur_col, cur_col + PIXEL_RATIO)

            if USE_HSV:
                curr_target_color = target_hsv[target_slice]
            else:
                curr_target_color = target[target_slice]

            hue, saturation, value = np.mean(curr_target_color, axis=0).mean(axis=0)
            iter_like_orders.append([palette, hue, saturation, value])
            # proc = Process(target=find_closes_image, args=(que, )
            # proc.start()

        results = pool.map(find_closes_image, iter_like_orders)

        "Replace ROI"
        for col_num, cur_col in enumerate(range(0, w, PIXEL_RATIO)):
            output_slice = (
                    slice(row_num * AVATAR_SIZE, row_num * AVATAR_SIZE + AVATAR_SIZE),
                    slice(col_num * AVATAR_SIZE, col_num * AVATAR_SIZE + AVATAR_SIZE)
            )

            roi = output[output_slice]
            found = results.pop(0)

            if found:
                replacer = cv2.imread(found, cv2.IMREAD_COLOR)
                replacer = imutils.resize(replacer, height=AVATAR_SIZE)
                try:
                    roi[:, :, :] = replacer
                except ValueError as err:
                    if ignore_image_size and fill_border_at_error and False:
                        last_h, last_w, _ = roi.shape
                        replacer = replacer[0:last_h, 0:last_w, :]
                        roi[:, :, :] = replacer
                    elif ignore_image_size:
                        pass
                    else:
                        print(f"{err}")
        timeend = time.time()
        duration = timeend - time0
        print(f"Row {cur_row} was executed in: {duration:>4.1f}s")
    return output


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


"Params"
USE_HSV = False
PIXEL_RATIO = 5
AVATAR_SIZE = 50
SAVE_EXT = "jpg"

"Input photo"
target_path = "src_images/space1.jpg"
name = str(os.path.basename(target_path)).split('.')[0]
output = get_mozaic(target_path, ignore_image_size=True)

"Find non overwriting path"
out_path = f"output/mozaic/{name}-000.{SAVE_EXT}"
num = 0
while os.path.isfile(out_path):
    num += 1
    out_path = f"output/mozaic/{name}-{num:>03}.{SAVE_EXT}"

"Save picture"
cv2.imwrite(out_path, output)
print(f"Saved to: {out_path}")

"Preview"
preview = imutils.resize(output, height=500)
cv2.imshow("small preview", preview)
cv2.waitKey(10_000)
