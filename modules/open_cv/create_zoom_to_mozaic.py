from multiprocessing import Process, Queue, Pool
from PIL import Image
from io import BytesIO

import traceback
import imutils
import pickle
import random
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
            print(f"{func.__name__} was executed in {duration / 60:>6.2f}m")
        return result

    return wrapper


@time_it_dec
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

        # if USE_HSV:
        #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #     hue, saturation, value = np.mean(hsv, axis=0).mean(axis=0)
        # else:
        blue, green, red = np.mean(image, axis=0).mean(axis=0)
        palette[image_path] = {"r": red, "g": green, "b": blue}
    print(f"Palette size: {len(palette.keys())}")
    return palette


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


def find_closes_image(arguments):
    targ_blue, targ_green, targ_red, palette = arguments
    error = 255 ** 2
    best = None
    for path, inner_dict in palette.items():
        red, green, blue = inner_dict['r'], inner_dict['g'], inner_dict['g']
        cur_error = (targ_blue - blue) ** 2 + (targ_green - green) ** 2 + (targ_red - red) ** 2  # squared error
        if cur_error < error:

            error = cur_error
            best = path
    return best


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


@time_it_dec
def make_gif(image, cx=None, cy=None, ):
    FRAMES = 500
    name = "gif_zoom"
    height, width, _ = image.shape
    origin_x = width // 2
    origin_y = height // 2
    frame = scale_image_to_max_dim(image, OUTPUT_GIF_MAX_SIZE)
    out_height, out_width, _ = frame.shape
    # if height != width:
    #     if height > width:
    #         origin_width = 50
    #         origin_height = int(height / (width / 50))
    #     else:
    #         origin_width = int(width / (height / 50))
    #         origin_height = 50
    #
    # else:
    origin_width = out_width
    origin_height = out_height

    origin_x = origin_x - origin_width // 2
    origin_x = 0 if origin_x < 0 else origin_x
    origin_y = origin_y - origin_height // 2
    origin_y = 0 if origin_y < 0 else origin_y
    # print(f"Y: {origin_height}, X:{origin_width}")
    # print(f"Ending height: {height}, width: {width}")

    arr_x = np.linspace(origin_x, 0, FRAMES)
    arr_y = np.linspace(origin_y, 0, FRAMES)
    arr_height = np.linspace(out_height, height, FRAMES)
    arr_width = np.linspace(out_width, width, FRAMES)
    roi_slice = slice(origin_y, origin_y + out_height), \
                slice(origin_x, origin_x + out_width), \
                slice(None)

    frame = image[roi_slice]
    frame = scale_image_to_max_dim(frame, OUTPUT_GIF_MAX_SIZE)

    pil_image = image_array_to_pillow(frame)

    framerate = 1 / 120
    duration = 1 / framerate
    frames_list = [pil_image, pil_image, pil_image]

    for fr, x, y, w, h in zip(range(FRAMES), arr_x, arr_y, arr_width, arr_height):
        if 490 > fr > 10 and (fr % 2 or fr % 3 or fr % 7):
            continue
        x = round(x)
        y = round(y)
        w = round(w)
        h = round(h)
        # print(f"Frame: {fr}: xy {x} - {y}, w:{w}, h:{h}")
        roi_slice = slice(y, y + h), slice(x, x + w), slice(None)
        roi = image[roi_slice]

        # frame = scale_image_to_max_dim(roi, OUTPUT_GIF_MAX_SIZE)  # sometimes is missing 1 pixel width/height
        frame = cv2.resize(roi, (out_width, out_height))
        pil_image = image_array_to_pillow(frame)
        frames_list.append(pil_image)

    frames_list.append(pil_image)
    frames_list.append(pil_image)
    frames_list[0].save(f"output/gif/{name}-pow{PROC_POWER}-siz{OUTPUT_GIF_MAX_SIZE}.gif", format="GIF", append_images=frames_list[1:],
                        save_all=True, optimize=False, duration=duration, loop=0)
    print(f"Saved GIF.")


@time_it_dec
def get_mozaic(target, palette, ignore_image_size=True, fill_border_at_error=False):
    # target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    h, w, c = target.shape

    if (h % PIXEL_RATIO or w % PIXEL_RATIO) and not ignore_image_size:
        print(f"Invalid size, H:{h}%->{h % PIXEL_RATIO}, W:{w}->{w % PIXEL_RATIO}")
        sys.exit(0)

    output = np.zeros(
            (h * AVATAR_SIZE // PIXEL_RATIO, w * AVATAR_SIZE // PIXEL_RATIO, 3),
            dtype=np.uint8
    )

    pool = Pool(38)
    loop_range = len(range(0, h, PIXEL_RATIO))
    for row_num, cur_row in enumerate(range(0, h, PIXEL_RATIO)):

        time0 = time.time()
        "Make Queue"
        iter_like_orders = []
        for col_num, cur_col in enumerate(range(0, w, PIXEL_RATIO)):
            target_slice = slice(cur_row, cur_row + PIXEL_RATIO), slice(cur_col, cur_col + PIXEL_RATIO)

            # if USE_HSV:
            #     curr_target_color = target_hsv[target_slice]
            # else:
            curr_target_color = target[target_slice]

            blue, green, red = np.mean(curr_target_color, axis=0).mean(axis=0)
            iter_like_orders.append([blue, green, red, palette])
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
                    if ignore_image_size and fill_border_at_error:
                        last_h, last_w, _ = roi.shape
                        replacer = replacer[0:last_h, 0:last_w, :]
                        roi[:, :, :] = replacer
                    elif ignore_image_size:
                        pass
                    else:
                        print(f"{err}")
        timeend = time.time()
        duration = timeend - time0
        print(f"{row_num:>3} of {loop_range} was executed in: {duration:>4.1f}s")
    return output


def _(img_path):
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


def scale_image_to_max_dim(image, max_val):
    height, width, _ = image.shape

    if height > max_val or width > max_val:
        if height > width:
            image = imutils.resize(image, height=max_val)
        else:
            image = imutils.resize(image, width=max_val)
    return image


def load_palette():
    if os.path.isfile(PAL_PATH):
        with open(PAL_PATH, "rb") as fp:
            palette = pickle.load(fp)
    else:
        raise FileNotFoundError("Palette does not exists")
    # print(f"Palette loaded, size: {len(palette)}")  # size 1
    return palette


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
    for directory in os.listdir(PAL_DIR):
        cur_path = os.path.join(PAL_DIR, directory)
        if os.path.isdir(cur_path):
            arr = create_palette(cur_path)
            pal_dirs.update({directory: arr})

    with open(PAL_PATH, "wb") as fp:
        pickle.dump(pal_dirs, fp)


def make_mozaic_and_gif(target_path, selection=None):
    try:
        image = cv2.imread(target_path, cv2.IMREAD_COLOR)
        image = scale_image_to_max_dim(image, MAX_PIC_SIZE)
        height, width, _ = image.shape

        all_pallettes = load_palette()
        palette = select_palette(all_pallettes, selection)
        mozaic = get_mozaic(image, palette, fill_border_at_error=FILL_BORDER_WHEN_CROP)
        # cv2.imwrite(f"mozaic.png", mozaic)
        print(f"Mozaic size: {mozaic.shape}")
        make_gif(mozaic)

    except Exception as err:
        tb_text = traceback.format_exception(type(err), err, err.__traceback__)
        tb_text = ''.join(tb_text)
        print(f"Error during making mozaic and gif: {tb_text}")


def select_palette(all_palettes, selection=None):
    if selection:
        cur = [value for pal_name, value in all_palettes.items()
               for sel in selection if sel in pal_name]
        palette = {}
        for c in cur:
            palette.update({**c})

    else:
        palette = random.choice(all_palettes.values())
    print(f"Palette size: {len(palette)}")
    return palette


"Params"

OUTPUT_GIF_MAX_SIZE = 500
PROC_POWER = 200
PIXEL_RATIO = round(OUTPUT_GIF_MAX_SIZE/PROC_POWER)
print(f"PIXEL ratio: {PIXEL_RATIO}")
print(f"Output size: {OUTPUT_GIF_MAX_SIZE}")
print(f"Processing power: {PROC_POWER}")
# USE_HSV = False
# PIXEL_RATIO = 3
AVATAR_SIZE = 50
# MAX_PIC_SIZE = 600
MAX_PIC_SIZE = OUTPUT_GIF_MAX_SIZE
SAVE_EXT = "jpg"
FILL_BORDER_WHEN_CROP = True
OUTPUT_GIF_MAX_SIZE = MAX_PIC_SIZE  # This will reduce aliasing
target_path = "src_images/shapes.png"
# PROC_POWER = MAX_PIC_SIZE / PIXEL_RATIO


"Consts"
PAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "palette"))
PAL_PATH = os.path.join(PAL_DIR, "palettes.pickle")

"Run"
make_mozaic_and_gif(target_path, selection=["open"])
# recreate_all_palettes()
# load_palette()
