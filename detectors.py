import numpy as np
import cv2

debug = 0
# Надзвичайно важливий параметр - радіус відстежуваного об'єкта
# Для video2 - 80
# Для video4 - 20
blob_radius_thresh = 20


class Detectors(object):

    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if debug == 1:
            cv2.imshow('gray', gray)

        fgmask = self.fgbg.apply(gray)

        if debug == 0:
            cv2.imshow('bgsub', fgmask)

        edges = cv2.Canny(fgmask, 50, 190, 3)

        if debug == 1:
            cv2.imshow('Edges', edges)

        ret, thresh = cv2.threshold(edges, 127, 255, 0)

        contours, hierarchy = cv2.findContours(thresh,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        if debug == 0:
            cv2.imshow('thresh', thresh)

        centers = []

        for cnt in contours:
            try:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                centroid = (int(x), int(y))
                radius = int(radius)
                if radius > blob_radius_thresh:
                    cv2.circle(frame, centroid, radius, (0, 255, 0), 2)
                    b = np.array([[x], [y]])
                    centers.append(np.round(b))
            except ZeroDivisionError:
                pass

        cv2.imshow('Track obj', frame)

        return centers
