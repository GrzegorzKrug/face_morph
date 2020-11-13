from PIL import Image
from io import BytesIO

import numpy as np
import cv2


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
