import cv2 as cv
import imutils
import numpy as np
import copy

from imutils import contours


class Coords:
    x = 0
    y = 0
    h = 0
    w = 0

    def __init__(self, x, y, h, w):
        self.x = x
        self.y = y
        self.h = h
        self.w = w


def cut_photo(photo_name):
    src = cv.imread(photo_name)
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

    lowerColor = np.array([151, 24, 255])
    upperColor = np.array([179, 255, 255])
    mask = cv.inRange(hsv, lowerColor, upperColor)
    result = cv.bitwise_and(src, src, mask=mask)

    result_grayImage = cv.blur(cv.cvtColor(result, cv.COLOR_BGR2GRAY), (3, 3))
    result_grayImage = cv.threshold(result_grayImage, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

    threshold = 100
    canny_output = cv.Canny(result_grayImage, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    cv.imshow("zefzefzf", result_grayImage)
    cv.waitKey(0)
    pt1 = Coords(9999, 9999, 0, 0)
    pt2 = Coords(-9999, 9999, 0, 0)
    pt3 = Coords(9999, 9999, 0, 0)
    pt4 = Coords(-9999, -9999, 0, 0)

    for i in range(len(contours)):
        x, y, w, h = boundRect[i]

        if pt1.x > x:
            pt1.x = x

        if pt1.y > y:
            pt1.y = y
            pt1.h = h
            pt1.w = w

        if pt2.x < x and pt2.y > y:
            pt2.x = x
            pt2.y = y
            pt2.h = h
            pt2.w = w

        if pt3.x > x and pt3.y > y:
            pt3.x = x
            pt3.y = y
            pt3.h = h
            pt3.w = w

        if pt4.x < x and pt4.y < y:
            pt4.x = x
            pt4.y = y
            pt4.h = h
            pt4.w = w

    pt2.x += round(pt2.w * 4)
    pt2.y -= round(pt2.w * 4)

    pt1.y -= round(pt1.h * 0.5)
    pt1.x += round(pt1.w * 0.5)

    pt3.y += pt3.h
    pt3.x -= round(pt3.w * 1.5)

    pt4.y += pt4.h
    pt4.x += pt4.w * 5

    newImage = copy.copy(src)
    cv.circle(newImage, (pt1.x, pt1.y), 5, (0, 0, 255), -1)  # Red
    cv.circle(newImage, (pt2.x, pt2.y), 5, (255, 0, 0), -1)  # Blue
    cv.circle(newImage, (pt3.x, pt3.y), 5, (0, 255, 0), -1)  # Green
    cv.circle(newImage, (pt4.x, pt4.y), 5, (0, 255, 255), -1)  # Yellow
    cv.imshow('Contours', newImage)
    cv.waitKey(0)

    pts1 = np.float32(
        [[pt1.x, pt1.y], [pt2.x, pt2.y], [pt3.x, pt3.y], [pt4.x, pt4.y]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 150], [500, 150]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    print("⚙ Image correctly processed ...")
    return cv.warpPerspective(src, matrix, (500, 150))


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


def treatImage(cropedImage):
    try:

        print("🚀 Starting numbers detection...")
        image = imutils.resize(cropedImage, 300)
        noirblanc = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        imagecontraste = cv.threshold(noirblanc, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        L_contours, t = cv.findContours(imagecontraste, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        L_contours_chiffre = []

        for cont in L_contours:
            if tuple(cont[cont[:, :, 1].argmin()][0])[1] < 50:
                (x, y, largeur, hauteur) = cv.boundingRect(cont)
                if hauteur > 20 and hauteur < 60:
                    L_contours_chiffre.append(cont)

        L_contours_chiffre = contours.sort_contours(L_contours_chiffre, "left-to-right")[0]
        L_chiffres = []

        for c in L_contours_chiffre:
            (x, y, largeur, hauteur) = cv.boundingRect(c)
            chiffrecoupe = imagecontraste[y:y + hauteur, x:x + largeur]

            (chiffrecoupeH, chiffrecoupeL) = chiffrecoupe.shape
            (segL, segH) = (int(chiffrecoupeL * 0.35), int(chiffrecoupeH * 0.25))
            segLcentre = int(chiffrecoupeH * 0.05)

            segments = [
                ((0, 0), (largeur, segH)),  # Segment Haut
                ((0, 0), (segL, hauteur // 2)),  # Segment Haut Gauche
                ((largeur - segL, 0), (largeur, hauteur // 2)),  # Segment Haut Droit
                ((0, (hauteur // 2) - segLcentre), (largeur, (hauteur // 2) + segLcentre)),  # Segment Centre
                ((0, hauteur // 2), (segL, hauteur)),  # Segment Bas Gauche
                ((largeur - segL, hauteur // 2), (largeur, hauteur)),  # Segment Bas Droit
                ((0, hauteur - segH), (largeur, hauteur))  # Segment Bas
            ]
            segmentsOn = [0] * len(segments)
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                segmentCoord = chiffrecoupe[yA:yB, xA:xB]
                aire = cv.countNonZero(segmentCoord)
                airetotal = (xB - xA) * (yB - yA)

                if aire / float(airetotal) > 0.4:
                    segmentsOn[i] = 1
            if largeur > 5 and largeur < 20:
                chiffre = 1
            else:
                chiffre = DICTIONNAIRE_CHIFFRES[tuple(segmentsOn)]
            L_chiffres.append(chiffre)
        return L_chiffres
    except:
        return "null"


def complete_processing(source_image):
    return treatImage(cut_photo(source_image))
