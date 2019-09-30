import cv2 as cv
import imutils
import numpy as np
from scipy.constants import sigma

src = cv.imread("webcam_12-Sep-2019_13-17-07.jpg")
hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

image = imutils.resize(src, height=500)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
v = np.median(blurred)
edged = cv.Canny(blurred, int(max(0, (1.0 - sigma) * v)), int(min(255, (1.0 + sigma) * v)))
cv.imshow("ergerg", edged)
cv.waitKey(100)