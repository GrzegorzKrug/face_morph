from tkinter import (
    Frame, Grid, Canvas,
    Button, Label,
    Scale,
    messagebox)

from PIL import Image, ImageTk
from collections import deque
import tkinter as tk
import numpy as np
import time
import sys
import cv2

LARGE_FONT = ("Verdana", 12)


class CameraWindow(Frame):
    def __init__(self, root=0, video_source=None):
        Frame.__init__(self)
        self.root = root
        self.cam = video_source
        self.pic = Canvas(root, width=1000, height=800)
        self.pic.pack(side='top', fill='both', expand=True)
        self.memory = deque(maxlen=30 * 1)
        self.refresh_interval = 20
        self.config = {
                'grayscale': False,
                'gc_on': False,
                'xmin': 0,
                'xmax': 0,
                'ymin': 0,
                'ymax': 0,
                'zmin': 0,
                'zmax': 0,
                'draw_rect': False,
        }

        button_frame = Frame(root)
        button_frame.pack(fill='both', expand=True)

        add_button(button_frame, f'GrayScale', 'black', row=0, col=0, sticky="wse").configure(
                command=lambda: self.toggle_config('grayscale'))
        add_button(button_frame, f'Draw Rectangle', 'black', row=0, col=1, sticky="wse").configure(
                command=lambda: self.toggle_config('draw_rect'))
        add_button(button_frame, f'Foreground', 'black', row=0, col=2, sticky="wse").configure(
                command=lambda: self.toggle_config('gc_on'))
        add_button(button_frame, f'{3 + 1}', 'black', row=0, col=3, sticky="wse").configure(
                command=lambda: print(3 + 1))
        add_button(button_frame, f'{4 + 1}', 'black', row=0, col=4, sticky="wse").configure(
                command=lambda: print(4 + 1))
        add_button(button_frame, f'{5 + 1}', 'black', row=0, col=5, sticky="wse").configure(
                command=lambda: print(5 + 1))
        add_button(button_frame, f'{6 + 1}', 'black', row=0, col=6, sticky="wse").configure(
                command=lambda: print(6 + 1))
        add_button(button_frame, f'{7 + 1}', 'black', row=0, col=7, sticky="wse").configure(
                command=lambda: print(7 + 1))
        add_button(button_frame, f'{8 + 1}', 'black', row=0, col=8, sticky="wse").configure(
                command=lambda: print(8 + 1))
        add_button(button_frame, f'{9 + 1}', 'black', row=0, col=9, sticky="wse").configure(
                command=lambda: print(9 + 1))

        quit = add_button(button_frame, f'Quit', 'black', row=0, col=9, sticky="wse")
        quit.configure(
                command=lambda: sys.exit(0))

        for _col in range(10):
            button_frame.grid_columnconfigure(_col, weight=1)

        button_frame.grid_rowconfigure(0, weight=1)

        self.scale1 = Scale(root, label="Refresh", from_=10, to=500, command=self.set_refresh_interval)
        self.scale1.pack(side="left")

        self.scale2 = Scale(root, label="X-Min", to=255)
        self.scale2.pack(side="left")
        self.scale2.configure(command=lambda val: self.set_int_value('xmin', val))

        self.scale4 = Scale(root, label="X-Max", to=255)
        self.scale4.pack(side="left")
        self.scale4.configure(command=lambda val: self.set_int_value('xmax', val))

        self.scale3 = Scale(root, label="Y-Min", to=255)
        self.scale3.pack(side="left")
        self.scale3.configure(command=lambda val: self.set_int_value('ymin', val))

        self.scale5 = Scale(root, label="Y-Max", to=255)
        self.scale5.pack(side="left")
        self.scale5.configure(command=lambda val: self.set_int_value('ymax', val))

        self.scale3 = Scale(root, label="Z-Min", to=255)
        self.scale3.pack(side="left")
        self.scale3.configure(command=lambda val: self.set_int_value('zmin', val))

        self.scale5 = Scale(root, label="Z-Max", to=255)
        self.scale5.pack(side="left")
        self.scale5.configure(command=lambda val: self.set_int_value('zmax', val))

        self.scale8 = Scale(root, label="_8")
        self.scale8.pack(side="left")

        self.update()

    def set_refresh_interval(self, new_value):
        self.refresh_interval = int(new_value)

    def toggle_config(self, param):
        self.config.update({param: self.config[param] ^ True})
        print(f"{param} is now: {self.config[param]}")

    def set_int_value(self, param, new_value):
        self.config.update({param: int(new_value)})
        print(f"{param} is now: {self.config[param]}")

    def set_pic_size(self, width, height):
        self.pic.configure(width=width, height=height)

    def update(self):
        ret, frame = self.cam.get_frame()
        image = self.process_image(frame)
        photo = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=photo)
        self.memory.append(photo)  # reference to photo must persist
        self.pic.create_image(0, 0, image=photo, anchor='nw')
        self.root.after(self.refresh_interval, self.update)

    def process_image(self, image):
        """Returns Tkinter Canvas photo object"""
        xmin = self.config['xmin']
        xmax = self.config['xmax']
        ymin = self.config['ymin']
        ymax = self.config['ymax']
        zmin = self.config['zmin']
        zmax = self.config['zmax']

        image = self.strip_color(image, xmin, xmax, ymin, ymax, zmin, zmax)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def strip_color(self, image, hmin, hmax, smin, smax, vmin, vmax):
        frame_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_orange = np.array([hmin, smin, vmin])
        upper_orange = np.array([hmax, smax, vmax])

        mask_alpha = cv2.inRange(frame_hsv, lower_orange, upper_orange)
        mask = mask_alpha < 1

        frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
        return frame

    def grab_cut(self, image, rect):
        mask = np.zeros(image.shape[:2], np.uint8)

        bdgModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(image, mask, rect, bdgModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')[:, :, np.newaxis]
        image = image * mask2

        return image


def add_button(_frame, text, fg, side=None, fill=None,
               row=0, col=0, sticky=None):
    but = Button(_frame, text=text, fg=fg)
    but.grid(row=row, column=col, sticky=sticky)
    return but


class MyCameraCapture:
    def __init__(self, video_source=0):
        print(f"Opened camera src")

        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, frame
            else:
                return ret, None
        else:
            return None, None

    # Release the video source when the object is destroyed
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def __del__(self):
        print(f"deleting")
        if self.vid.isOpened():
            self.vid.release()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Camera View")

    with MyCameraCapture(0) as c1:
        cam_win = CameraWindow(root, c1)
        width, height = c1.width, c1.height
        cam_win.set_pic_size(width=width, height=height)
        cam_win.cam = c1
        # cam_win.update()
        root.mainloop()
