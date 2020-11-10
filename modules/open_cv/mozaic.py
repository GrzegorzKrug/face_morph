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
        print(f"{func.__name__} was executed in {duration:>6.2f}s")
        return result

    return wrapper


@time_it_dec
def create_palette():
    palette = {}
    all_images = glob.glob("square_stamps/*//*.png", recursive=True)
    all_images += glob.glob("square_stamps/*//*.jpg", recursive=True)

    for image_path in all_images:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        height, width, channels = image.shape

        if height < 15 or width < 15:
            print(f"This image is too small: {height:>4}, {width:>4} - {image_path}")
            continue
        elif height != width:
            print(f"This image is not squared: {height:>4}, {width:>4} - {image_path}")
            image = squareify_stamp(image_path)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = np.mean(hsv, axis=0).mean(axis=0)
        palette[image_path] = {"h": hue, "sat": saturation, "val": value}

    return palette


# @time_it_dec
def find_closes_image(_palette, targ_h, targ_s, targ_v):
    error = 255 ** 2
    best = None
    for path, inner_dict in _palette.items():
        h, s, v = inner_dict.values()
        cur_error = abs(targ_h - h) + abs(targ_s - s) + abs(targ_v - v)  # Linear error
        # cur_error = (targ_h - h) ** 2 + (targ_s - s) ** 2 + (targ_v - v) ** 2  # squared error
        if cur_error < error:

            error = cur_error
            best = path
    # print(f"{targ_h}{targ_s}{targ_v}", error, best)
    return best


def get_mozaic(targ_path):
    target = cv2.imread(targ_path, cv2.IMREAD_COLOR)
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    h, w, c = target.shape
    PIX_SIZE = 5
    palette = create_palette()
    if h % PIX_SIZE or w % PIX_SIZE:
        print("Invalid size")
        sys.exit(0)

    output = np.zeros((h, w, 3), dtype=np.uint8)

    for cur_row in range(0, h, PIX_SIZE):
        for cur_col in range(0, w, PIX_SIZE):
            current_roi_slice = slice(cur_row, cur_row + PIX_SIZE), slice(cur_col, cur_col + PIX_SIZE)
            roi = output[current_roi_slice]
            curr_target_hsv = target_hsv[current_roi_slice]

            hue, saturation, value = np.mean(curr_target_hsv, axis=0).mean(axis=0)
            found = find_closes_image(palette, hue, saturation, value)
            if found:
                replacer = cv2.imread(found, cv2.IMREAD_COLOR)
                replacer = imutils.resize(replacer, width=PIX_SIZE)
                roi[:, :, :] = replacer
    return output


def squareify_stamp(img_path):
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


target_path = "src_images/salema-beach.jpg"
output = get_mozaic(target_path)

cv2.imshow("Output", output)
out_path = "output/mozaic-0.png"
num = 0
while os.path.isfile(out_path):
    num += 1
    print(num)
    out_path = f"output/mozaic-{num}.png"

cv2.imwrite(out_path, output)
print(f"Saved to: {out_path}")
cv2.waitKey()
