from copy import deepcopy

import cv2 as cv
import src.Utilities as utils


def nothing(x):
    pass


class HoughCirclesTrackbar:
    def __init__(self, title, image, params):
        self.params = params
        self.image = image
        self.title = title

    def init_trackbars(self):
        cv.namedWindow(self.title)

        cv.createTrackbar('dp', self.title, self.params['dp'], 30, nothing)
        cv.createTrackbar('minDist', self.title, self.params['minDist'], 100, nothing)
        cv.createTrackbar('param1', self.title, self.params['param1'], 255, nothing)
        cv.createTrackbar('param2', self.title, self.params['param2'], 255, nothing)
        cv.createTrackbar('minRadius', self.title, self.params['minRadius'], 100, nothing)
        cv.createTrackbar('maxRadius', self.title, self.params['maxRadius'], 100, nothing)

    def update_trackbars(self):
        for key, value in self.params.items():
            if key == 'dp':
                self.params[key] = cv.getTrackbarPos(key, self.title) / 10
            self.params[key] = cv.getTrackbarPos(key, self.title)

    def apply_hough(self, image):
        circles = cv.HoughCircles(image,
                                  method=cv.HOUGH_GRADIENT,
                                  dp=1,
                                  minDist=self.params['minDist'],
                                  param1=self.params['param1'],
                                  param2=self.params['param2'],
                                  minRadius=self.params['minRadius'],
                                  maxRadius=self.params['maxRadius'])
        return circles

    def run(self):
        self.init_trackbars()

        while True:
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Program terminated")
                break

            aux_image = deepcopy(self.image)

            self.update_trackbars()

            if not self.valid_params():
                print("Invalid paramteres!")
                continue

            circles = self.apply_hough(aux_image)

            utils.display_circles('Trackbar Circles', aux_image, circles)

        cv.destroyAllWindows()
        cv.waitKey(1)

    def valid_params(self):
        for key, value in self.params.items():
            if key == 'minRadius' or key == 'maxRadius':
                continue
            if value == 0:
                return False
        return True
