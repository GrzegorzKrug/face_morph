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
                'gc_xmin': 0,
                'gc_xmax': 0,
                'gc_ymin': 0,
                'gc_ymax': 0,
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

        self.scale2 = Scale(root, label="X-Min", to=1000)
        self.scale2.pack(side="left")
        self.scale2.configure(command=lambda val: self.set_int_value('gc_xmin', val))

        self.scale4 = Scale(root, label="X-Max", to=1000)
        self.scale4.pack(side="left")
        self.scale4.configure(command=lambda val: self.set_int_value('gc_xmax', val))

        self.scale3 = Scale(root, label="Y-Min", to=1000)
        self.scale3.pack(side="left")
        self.scale3.configure(command=lambda val: self.set_int_value('gc_ymin', val))

        self.scale5 = Scale(root, label="Y-Max", to=1000)
        self.scale5.pack(side="left")
        self.scale5.configure(command=lambda val: self.set_int_value('gc_ymax', val))

        self.scale6 = Scale(root, label="_6")
        self.scale6.pack(side="left")
        self.scale7 = Scale(root, label="_7")
        self.scale7.pack(side="left")
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
        photo = self.process_image(frame)
        self.memory.append(photo)  # reference to photo must persist
        self.pic.create_image(0, 0, image=photo, anchor='nw')

        self.root.after(self.refresh_interval, self.update)

    def process_image(self, image):
        """Returns Tkinter Canvas photo object"""

        # image = np.frombuffer(image, np.uint8)
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        rect = [
                self.config['gc_xmin'],
                self.config['gc_ymin'],
                self.config['gc_xmax'],
                self.config['gc_ymax'],
        ]
        if rect[0] > rect[2]:
            rect[0], rect[2] = rect[2], rect[0]
        if rect[1] > rect[3]:
            rect[3], rect[1] = rect[1], rect[3]

        if self.config['grayscale']:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.config["gc_on"]:
                image = self.grab_cut(image, rect)

        if self.config['draw_rect']:
            pt1 = (rect[0], rect[1])
            pt2 = (rect[2], rect[3])
            cv2.rectangle(image, pt1, pt2, (0, 255, 255), 3)

        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image=image)
        return image

    def grab_cut(self, image, rect):
        mask = np.zeros(image.shape[:2], np.uint8)

        bdgModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(image, mask, rect, bdgModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask2[:, :, np.newaxis]

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

    with MyCameraCapture() as c1:
        cam_win = CameraWindow(root, c1)
        width, height = c1.width, c1.height
        cam_win.set_pic_size(width=width, height=height)
        cam_win.cam = c1
        # cam_win.update()
        root.mainloop()
