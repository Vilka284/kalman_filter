import numpy as np
import cv2

debug = 0


class Detectors(object):

    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def Detect(self, frame):

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
        blob_radius_thresh = 80
        for cnt in contours:
            try:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                centeroid = (int(x), int(y))
                radius = int(radius)
                if radius > blob_radius_thresh:
                    cv2.circle(frame, centeroid, radius, (0, 255, 0), 2)
                    b = np.array([[x], [y]])
                    centers.append(np.round(b))
            except ZeroDivisionError:
                pass

        cv2.imshow('Track obj', frame)

        return centers
