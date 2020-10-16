import numpy as np
import glob
import cv2
import os

eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

avatars = glob.glob("avatars/*png")
avatars.sort()
export_dir = "exports"
export_dir = "exports"
face_count = 0
MAX_SIZE = 256
os.makedirs(export_dir, exist_ok=True)

for avatar in avatars:
    bgr_img = cv2.imread(avatar)
    width, height, channels = bgr_img.shape

    if width > MAX_SIZE or height > MAX_SIZE:
        width = MAX_SIZE if width > MAX_SIZE else width
        height = MAX_SIZE if height > MAX_SIZE else height

        dim = (width, height)
        bgr_img = cv2.resize(bgr_img, dim)

    gray_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image, 1.1, 3)
    for (x, y, w, h) in faces:
        face_count += 1
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(bgr_img, pt1, pt2, (0, 255, 55), 2)

    eyes = eye_cascade.detectMultiScale(gray_image, 1.2, 8)
    for (x, y, w, h) in eyes:
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(bgr_img, pt1, pt2, (255, 85, 0), 2)

    cv2.imwrite(os.path.join(export_dir, os.path.basename(avatar)), bgr_img)

print(f"Faces found: {face_count}")
