import imutils
import numpy as np
import cv2
import os

shapes = cv2.imread(f"src_images/shapes3.png", cv2.IMREAD_COLOR)
shapes = imutils.resize(shapes, width=1000)

gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 100)

contours, hierarhcy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue
    cv2.polylines(shapes, cnt, True, (30, 185, 155), 5)

    pts = cnt.reshape(-1, 2)
    x, y = np.mean(cnt, axis=0).astype('int')[0]

    cv2.putText(shapes, f"{area:>4.0f}", (x - 55, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))

# cv2.drawContours(shapes, contours, -1, (0, 50, 255), 5)

cv2.imshow("shapes", shapes)
# cv2.imshow("canny", canny)
cv2.waitKey()
