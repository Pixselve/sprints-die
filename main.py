from __future__ import print_function

import copy
from datetime import datetime

from math import sqrt

import cv2 as cv
import imutils
import numpy as np
import argparse
import random as rng

rng.seed(12345)
src = cv.imread("webcam_12-Sep-2019_13-17-07.jpg")
hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

# Convert image to gray and blur it

source_window = 'Source'


class Images:
    _grayImage = ""

    def __init__(self, grayImage):
        self._grayImage = grayImage

    @property
    def grayImage(self):
        return self._grayImage

    @grayImage.setter
    def grayImage(self, value):
        self._grayImage = value


image = Images(cv.blur(cv.cvtColor(src, cv.COLOR_BGR2GRAY), (3, 3)))


class Coords:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Pictures:
    pt1 = Coords(9999, 9999)
    pt2 = Coords(9999, -9999)
    pt3 = Coords(-9999, 9999)
    pt4 = Coords(-9999, -9999)

    def savePics(self):
        cropedImage = src[self.minCoords.y - 25:self.maxCoords.y + 25, self.minCoords.x - 25:self.maxCoords.x + 25]
        # pts1 = np.float32([[470, 206], [1479, 198], [32, 1122], [1980, 1125]])
        # pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])

        cv.imwrite("./images/image.png", cropedImage)


picture = Pictures()


def thresh_callback(val):
    threshold = val

    canny_output = cv.Canny(image.grayImage, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    newImage = copy.copy(src)

    filteredBoundingRect = []

    contours123 = imutils.grab_contours(contours)
    c = max(contours123, key=cv.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    cv.circle(newImage, extLeft, 5, (0, 0, 255), -1)
    cv.circle(newImage, extRight, 5, (0, 255, 255), -1)
    cv.circle(newImage, extTop, 5, (255, 0, 255), -1)
    cv.circle(newImage, extBot, 5, (255, 0, 0), -1)



    # for i in range(len(contours)):
    #     x, y, w, h = boundRect[i]
    #
    #     if picture.pt1.x > x and picture.pt1.y > y:
    #         picture.pt1.x = x
    #         picture.pt1.y = y
    #         print("PT1")
    #     if picture.pt3.x > x and picture.pt3.y < y:
    #         picture.pt3.x = x
    #         picture.pt3.y = y
    #         print("PT3")
    #     if picture.pt2.x < x and picture.pt2.y > y:
    #         picture.pt2.x = x
    #         picture.pt2.y = y
    #         print("PT2")
    #     if picture.pt4.x < x and picture.pt4.y < y:
    #         picture.pt4.x = x
    #         picture.pt4.y = y
    #         print("PT4")

    # cv.circle(newImage, (picture.pt1.x, picture.pt1.y), 5, (0, 0, 255), -1)
    # cv.circle(newImage, (picture.pt2.x, picture.pt2.y), 5, (0, 255, 255), -1)
    # cv.circle(newImage, (picture.pt3.x, picture.pt3.y), 5, (255, 0, 255), -1)
    # cv.circle(newImage, (picture.pt4.x, picture.pt4.y), 5, (255, 0, 0), -1)
    # if picture.pt1.x > x:
    #     picture.pt1.x = x
    # if picture.pt1.y > y:
    #     picture.pt1.y = y
    # if picture.maxCoords.x < x:
    #     picture.maxCoords.x = x
    # if picture.maxCoords.y < y:
    #     picture.maxCoords.y = y
    # cv.rectangle(newImage, (picture.minCoords.x - 25, picture.minCoords.y - 25),
    #              (picture.maxCoords.x + 25, picture.maxCoords.y + 25), (255, 0, 0), 2)
    # for i in range(len(contours)):
    #     x, y, w, h = boundRect[i]
    #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #     cv.rectangle(newImage, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

    cv.imshow('Contours', newImage)


def takePhotoCallback(val):
    picture.savePics()


# Trackabars window
color_window = 'Color selection'
cv.namedWindow(color_window)
cv.createTrackbar("Take photo", color_window, 0, 1, takePhotoCallback)


def color_callback(val):
    lowerColor = np.array([lowerTrackbar.hue, lowerTrackbar.saturation, lowerTrackbar.value])
    upperColor = np.array([upperTrackbar.hue, upperTrackbar.saturation, upperTrackbar.value])
    mask = cv.inRange(hsv, lowerColor, upperColor)
    result = cv.bitwise_and(src, src, mask=mask)
    cv.imshow(color_window, result)
    image.grayImage = cv.blur(cv.cvtColor(result, cv.COLOR_BGR2GRAY), (3, 3))
    thresh_callback(cv.getTrackbarPos('Canny thresh:', source_window))


class Trackbar:
    hue_label = ""
    saturation_label = ""
    value_label = ""

    def __init__(self, hue, saturation, value, vh, vs, vv):
        self.value_label = value
        self.hue_label = hue
        self.saturation_label = saturation
        cv.createTrackbar(self.hue_label, color_window, vh, 179, color_callback)
        cv.createTrackbar(self.saturation_label, color_window, vs, 255, color_callback)
        cv.createTrackbar(self.value_label, color_window, vv, 255, color_callback)

    @property
    def hue(self):
        return cv.getTrackbarPos(self.hue_label, color_window)

    @property
    def saturation(self):
        return cv.getTrackbarPos(self.saturation_label, color_window)

    @property
    def value(self):
        return cv.getTrackbarPos(self.value_label, color_window)


def nothing(x):
    pass


lowerTrackbar = Trackbar("Lower Hue", "Lower Saturation", "Lower Light", 0, 19, 255)
upperTrackbar = Trackbar("Upper Hue", "Upper Saturation", "Upper Light", 65, 255, 255)

color_callback("")
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100  # initial threshold
cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv.waitKey()


class Personne:
    prenom = ""
    nom = ""

    def direbonjour(self):
        print("Bonjour, je suis " + self.prenom)


Mael_prenom = "Mael"
Mael_nom = "Kerichard"

mael = Personne("Mael", "Kerichard")
mael.direbonjour()
