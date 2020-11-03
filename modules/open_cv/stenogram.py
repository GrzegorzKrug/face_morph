import numpy as np
import math
import cv2

bits = 2


# DROP_SECRET_BITS = 0xFF >> bits
# KEEP_BITS = 0xFF ^ DROP_SECRET_BITS
# CLEAR_IMAGE_BITS = 0xFF >> bits << bits


def print_bits(value, postfix=None, n=8):
    text = bin(value)[2:]
    print(f"{text:>0{n}} {postfix}")


def keep_higher_bits(image, bits):
    drop_bits = 0xFF >> bits
    keep_bits = 0xFF ^ drop_bits
    out = image & keep_bits
    return out


def keep_lower_bits(image, bits):
    drop_bits = 0xFF >> bits << bits
    keep_bits = 0xFF ^ drop_bits
    out = image & keep_bits
    return out


def cut_higher_bits(image, bits):
    drop_bits = 0xFF >> bits
    out = image & drop_bits
    return out


def cut_lower_bits(secret, bits):
    keep_bits = 0xFF >> bits << bits
    out = secret & keep_bits
    return out


def hide(picture, secret, bits):
    picture = picture.copy()
    h, w, c = secret.shape
    roi = picture[:h, :w, :]

    roi[:, :, :] = cut_lower_bits(roi, bits)
    secret = secret >> (8 - bits)
    roi[:, :, :] = roi | secret

    return picture


def get_hidden_pic(picture, bits):
    secret = keep_lower_bits(picture, bits)
    secret = secret << (8 - bits)

    return secret


avatar = cv2.imread("avatars/Youshisu.png", 1)
background = cv2.imread("src_images/philadelphia.jpg", 1)
out_path = "output/stenogram.png"

message = keep_higher_bits(avatar, bits)
stenogram = hide(background, message, bits)
hidden_decoded = get_hidden_pic(stenogram, bits)

cv2.imshow("message", message)
# cv2.imshow("background", background)
cv2.imshow("stenogram", stenogram)
cv2.imshow("original", background)
cv2.imshow("hidden_decoded", hidden_decoded)
cv2.imwrite(out_path, stenogram)
cv2.waitKey(15_000)
cv2.destroyAllWindows()
