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

# Convert image to gray and blur it

# source_window = 'Source'
from imutils import contours


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

    #     threshold = val
    #
    # canny_output = cv.Canny(image.grayImage, threshold, threshold * 2)
    #
    # contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #
    # contours_poly = [None] * len(contours)
    # boundRect = [None] * len(contours)
    # centers = [None] * len(contours)
    # radius = [None] * len(contours)
    # for i, c in enumerate(contours):
    #     contours_poly[i] = cv.approxPolyDP(c, 3, True)
    #     boundRect[i] = cv.boundingRect(contours_poly[i])
    #     centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    #
    # newImage = copy.copy(src)
    #
    # for i in range(len(contours)):
    #     x, y, w, h = boundRect[i]
    #     if picture.pt1.x > x:
    #         picture.pt1.x = x + w * 3
    #
    #     if picture.pt1.y > y:
    #         picture.pt1.y = y - h
    #
    #     if picture.pt2.x < x and picture.pt2.y > y:
    #         picture.pt2.x = x + w * 3
    #         picture.pt2.y = y - h * 5
    #
    #     if picture.pt3.x > x and picture.pt3.y > y:
    #         picture.pt3.x = x - w * 3
    #         picture.pt3.y = y + h * 2
    #
    #     if picture.pt4.x < x and picture.pt4.y < y:
    #         picture.pt4.x = x + w * 2
    #         picture.pt4.y = y + h * 2
    #
    # cv.circle(newImage, (picture.pt1.x, picture.pt1.y), 5, (0, 0, 255), -1)  # Red
    # cv.circle(newImage, (picture.pt2.x, picture.pt2.y), 5, (255, 0, 0), -1)  # Blue
    # cv.circle(newImage, (picture.pt3.x, picture.pt3.y), 5, (0, 255, 0), -1)  # Green
    # cv.circle(newImage, (picture.pt4.x, picture.pt4.y), 5, (0, 255, 255), -1)  # Yellow
    #
    # cv.imshow('Contours', newImage)


class Coords:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Pictures:
    pt1 = Coords(9999, 9999)
    pt2 = Coords(-9999, 9999)
    pt3 = Coords(9999, 9999)
    pt4 = Coords(-9999, -9999)

    def getPic(self, source):
        pts1 = np.float32(
            [[self.pt1.x, self.pt1.y], [self.pt2.x, self.pt2.y], [self.pt3.x, self.pt3.y], [self.pt4.x, self.pt4.y]])
        pts2 = np.float32([[0, 0], [500, 0], [0, 150], [500, 150]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        result = cv.warpPerspective(source, matrix, (500, 150))
        resultBluredAndGray = cv.blur(cv.cvtColor(result, cv.COLOR_BGR2GRAY), (3, 3))
        return result


# picture = Pictures()



def takePhotoCallback(val):
    pass


# Trackabars window
# color_window = 'Color selection'
# cv.namedWindow(color_window)
# cv.createTrackbar("Take photo", color_window, 0, 1, takePhotoCallback)


def color_callback(val):
    pass


def nothing(x):
    pass


# color_callback("")
# cv.namedWindow(source_window)
# cv.imshow(source_window, src)
# max_thresh = 255
# thresh = 100  # initial threshold
# cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
# thresh_callback(thresh)

# cv.waitKey()


# Trackabars window
color_window = 'Color selection'
cv.namedWindow(color_window)
cv.createTrackbar("Take photo", color_window, 0, 1, takePhotoCallback)

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


DICTIONNAIRE_CHIFFRES = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

lowerTrackbar = Trackbar("Lower Hue", "Lower Saturation", "Lower Light", 0, 19, 255)
upperTrackbar = Trackbar("Upper Hue", "Upper Saturation", "Upper Light", 65, 255, 255)


################################################################### PARTIE II ##################################################################
def reco(img):
    picture = Pictures()
    src = cv.imread(img)
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    image = Images(cv.blur(cv.cvtColor(src, cv.COLOR_BGR2GRAY), (3, 3)))
    lowerColor = np.array([lowerTrackbar.hue, lowerTrackbar.saturation, lowerTrackbar.value])
    upperColor = np.array([upperTrackbar.hue, upperTrackbar.saturation, upperTrackbar.value])
    mask = cv.inRange(hsv, lowerColor, upperColor)
    result = cv.bitwise_and(src, src, mask=mask)
    image.grayImage = cv.blur(cv.cvtColor(result, cv.COLOR_BGR2GRAY), (3, 3))

    threshold = 100

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

    newImage = copy.copy(src)

    for i in range(len(contours)):
        x, y, w, h = boundRect[i]
    if picture.pt1.x > x:
        picture.pt1.x = x

    if picture.pt1.y > y:
        picture.pt1.y = y

    if picture.pt2.x < x and picture.pt2.y > y:
        picture.pt2.x = x
        picture.pt2.y = y

    if picture.pt3.x > x and picture.pt3.y > y:
        picture.pt3.x = x
        picture.pt3.y = y

    if picture.pt4.x < x and picture.pt4.y < y:
        picture.pt4.x = x
        picture.pt4.y = y

        # for i in range(len(contours)):
        #     x, y, w, h = boundRect[i]
        #     if picture.pt1.x > x:
        #         picture.pt1.x = x + w * 3
        #
        #     if picture.pt1.y > y:
        #         picture.pt1.y = y - h
        #
        #     if picture.pt2.x < x and picture.pt2.y > y:
        #         picture.pt2.x = x + w * 3
        #         picture.pt2.y = y - h * 5
        #
        #     if picture.pt3.x > x and picture.pt3.y > y:
        #         picture.pt3.x = x - w * 3
        #         picture.pt3.y = y + h * 2
        #
        #     if picture.pt4.x < x and picture.pt4.y < y:
        #         picture.pt4.x = x + w * 2
        #         picture.pt4.y = y + h * 2

        cv.circle(newImage, (picture.pt1.x, picture.pt1.y), 5, (0, 0, 255), -1)  # Red
        cv.circle(newImage, (picture.pt2.x, picture.pt2.y), 5, (255, 0, 0), -1)  # Blue
        cv.circle(newImage, (picture.pt3.x, picture.pt3.y), 5, (0, 255, 0), -1)  # Green
        cv.circle(newImage, (picture.pt4.x, picture.pt4.y), 5, (0, 255, 255), -1)  # Yellow

        cv.imshow('Contours', newImage)
        recognizeNumbers(picture.getPic(src))


def recognizeNumbers(imgAfterTreatment):
    cv.imshow("Output", imgAfterTreatment)
    cv.waitKey(0)
    image = imutils.resize(imgAfterTreatment, 300)
    noirblanc = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imagecontraste = cv.threshold(noirblanc, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    L_contours, t = cv.findContours(imagecontraste, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    L_contours_chiffre = []

    for cont in L_contours:
        if tuple(cont[cont[:, :, 1].argmin()][0])[1] < 50:
            (x, y, w, h) = cv.boundingRect(cont)
            if h > 20 and h < 60:
                L_contours_chiffre.append(cont)

    L_contours_chiffre = contours.sort_contours(L_contours_chiffre, "left-to-right")[0]
    L_chiffres = []

    for c in L_contours_chiffre:
        (x, y, w, h) = cv.boundingRect(c)
        roi = imagecontraste[y:y + h, x:x + w]

        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.35), int(roiH * 0.25))
        dHC = int(roiH * 0.05)

        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)),  # top-left
            ((w - dW, 0), (w, h // 2)),  # top-right
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),  # bottom-left
            ((w - dW, h // 2), (w, h)),  # bottom-right
            ((0, h - dH), (w, h))  # bottom
        ]
        on = [0] * len(segments)
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            segROI = roi[yA:yB, xA:xB]
            total = cv.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)

            if total / float(area) > 0.4:
                on[i] = 1
        if w > 5 and w < 20:
            chiffre = 1
        else:
            chiffre = DICTIONNAIRE_CHIFFRES[tuple(on)]
        print(chiffre)
        L_chiffres.append(chiffre)


reco("webcam_12-Sep-2019_13-17-07.jpg")
