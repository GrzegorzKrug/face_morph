from PIL import Image
from conversion import image_array_to_pillow

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


def on_bar_change(*args):
    pass
    # print(f"Args: {args}")


def big_picture_viewer():
    max_y, max_x, _ = image.shape
    cv2.imshow("ROI", np.zeros((10, 10, 3)))
    cv2.namedWindow("SLIDERS")
    cv2.createTrackbar("ROI_X", "SLIDERS", 0, max_x, on_bar_change)
    cv2.createTrackbar("ROI_Y", "SLIDERS", 0, max_y, on_bar_change)
    cv2.createTrackbar("ROI_SIZE", "SLIDERS", 800, 5_000, on_bar_change)

    while True:
        key = cv2.waitKey(100)

        if key == ord("q"):
            break

        x = cv2.getTrackbarPos("ROI_X", "SLIDERS")
        y = cv2.getTrackbarPos("ROI_Y", "SLIDERS")
        size = cv2.getTrackbarPos("ROI_SIZE", "SLIDERS")
        roi_slice = slice(y, y + size), slice(x, x + size), slice(None)
        roi = image[roi_slice]

        h, w, c = roi.shape
        if h > 800:
            try:
                preview = imutils.resize(roi, height=800, width=800)
            except Exception as err:
                print(f"Resize error: {err}")
                continue

        else:
            preview = roi

        try:
            cv2.imshow("ROI", preview)
        except Exception as err:
            print(f"Show error: {err}")
            cv2.imshow("ROI", np.zeros((100, 100, 3)))


def make_video():
    FRAMES = 500
    OUTPUT_HEIGHT = 500
    OUTPUT_WIDTH = 800

    origin_x = 21_300 - 15
    origin_y = 17_250
    origin_width = 80
    origin_height = 50

    arr_x = np.linspace(origin_x, 0, FRAMES)
    arr_y = np.linspace(origin_y, 0, FRAMES)
    arr_height = np.linspace(origin_height, HEIGHT, FRAMES)
    arr_width = np.linspace(origin_width, WIDTH, FRAMES)

    roi_slice = slice(origin_y, origin_y + origin_height), \
                slice(origin_x, origin_x + origin_width), \
                slice(None)
    frame = image[roi_slice]
    frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    framerate = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    path = "output/mp4/video.mp4"
    writer = cv2.VideoWriter(path, fourcc, framerate, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    for x in range(framerate):
        writer.write(frame)

    for fr, x, y, w, h in zip(range(FRAMES), arr_x, arr_y, arr_width, arr_height):
        if 490 > fr > 20 and (fr % 2 or fr % 3):
            continue

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        roi_slice = slice(y, y + h), slice(x, x + w), slice(None)
        roi = image[roi_slice]
        # print(roi_slice)

        # frame = imutils.resize(roi, height=OUTPUT_HEIGHT, )
        frame = cv2.resize(roi, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        print(f"Frame: {fr}, {frame.shape}")
        writer.write(frame)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(10)

    for x in range(framerate):
        writer.write(frame)
    writer.release()
    print(f"Saved video.")


def make_gif():
    FRAMES = 500
    OUTPUT_HEIGHT = 300
    origin_x = 14650
    origin_y = 10500

    height, width, _ = image.shape
    if height != width:
        if height > width:
            origin_width = 50
            origin_height = int(height / (width / 50))
        else:
            origin_width = int(width / (height / 50))
            origin_height = 50

    else:
        origin_width = 50
        origin_height = 50

    origin_x = origin_x - (origin_width - 50) // 2
    origin_y = origin_y - (origin_height - 50) // 2
    print(origin_height, origin_width)

    arr_x = np.linspace(origin_x, 0, FRAMES)
    arr_y = np.linspace(origin_y, 0, FRAMES)
    arr_height = np.linspace(origin_height, HEIGHT, FRAMES)
    arr_width = np.linspace(origin_width, WIDTH, FRAMES)

    roi_slice = slice(origin_y, origin_y + origin_height), \
                slice(origin_x, origin_x + origin_width), \
                slice(None)
    frame = image[roi_slice]
    frame = imutils.resize(frame, height=OUTPUT_HEIGHT)
    pil_image = image_array_to_pillow(frame)

    framerate = 1 / 120
    duration = 1 / framerate
    frames_list = [pil_image, pil_image, pil_image]

    for fr, x, y, w, h in zip(range(FRAMES), arr_x, arr_y, arr_width, arr_height):
        if 490 > fr > 10 and (fr % 2 or fr % 3 or fr % 4):
            continue
        print(f"Frame: {fr}")
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        roi_slice = slice(y, y + h), slice(x, x + w), slice(None)
        roi = image[roi_slice]
        # print(roi_slice)

        frame = imutils.resize(roi, height=OUTPUT_HEIGHT)
        pil_image = image_array_to_pillow(frame)
        frames_list.append(pil_image)

    frames_list.append(pil_image)
    frames_list.append(pil_image)
    frames_list[0].save(f"output/gif/{name}.gif", format="GIF", append_images=frames_list[1:],
                        save_all=True, optimize=False, duration=duration, loop=0)
    print(f"Saved GIF.")


im_path = "output/mozaic/cat-000.jpg"
# im_path = "output/mozaic/space1-002.jpg"
name = os.path.basename(im_path).split(".")[0]
image = cv2.imread(im_path)
HEIGHT, WIDTH, _ = image.shape

make_gif()

# big_picture_viewer()
