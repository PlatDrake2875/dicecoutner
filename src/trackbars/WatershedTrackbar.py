import cv2 as cv
import numpy as np


def nothing(x):
    pass


class WatershedTrackbar:
    def __init__(self, title, image, params):
        self.image = image
        self.title = title
        self.params = params

    def init_trackbars(self):
        cv.namedWindow(self.title)
        cv.createTrackbar('Threshold Value', self.title, self.params['threshold_value'], 255, nothing)
        cv.createTrackbar('Morph Kernel Size', self.title, self.params['morph_kernel_size'], 20, nothing)
        cv.createTrackbar('Opening Iterations', self.title, self.params['opening_iterations'], 10, nothing)
        cv.createTrackbar('Dilation Iterations', self.title, self.params['dilation_iterations'], 10, nothing)

    def update_trackbars(self):
        self.params['threshold_value'] = cv.getTrackbarPos('Threshold Value', self.title)
        self.params['morph_kernel_size'] = max(1, cv.getTrackbarPos('Morph Kernel Size',
                                                                    self.title) // 2 * 2 + 1)  # Ensure odd size
        self.params['opening_iterations'] = cv.getTrackbarPos('Opening Iterations', self.title)
        self.params['dilation_iterations'] = cv.getTrackbarPos('Dilation Iterations', self.title)

    def apply_watershed(self):
        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
            bgr_image = cv.cvtColor(self.image, cv.COLOR_GRAY2BGR)
        else:
            bgr_image = self.image.copy()

        gray = self.image
        ret, thresh = cv.threshold(gray, self.params['threshold_value'], 255, cv.THRESH_BINARY)

        kernel = np.ones((self.params['morph_kernel_size'], self.params['morph_kernel_size']), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=self.params['opening_iterations'])
        opening = cv.cvtColor(opening, cv.COLOR_BGR2GRAY) if len(opening.shape) == 3 else opening

        sure_bg = cv.dilate(opening, kernel, iterations=self.params['dilation_iterations'])

        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

        ret, sure_fg = cv.threshold(dist_transform, 0.8 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        unknown = cv.subtract(sure_bg, sure_fg)

        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers32 = np.int32(markers)
        cv.watershed(bgr_image, markers32)
        boundary_mask = np.zeros_like(bgr_image, dtype=np.uint8)
        boundary_mask[markers32 == -1] = [255, 255, 255]

        dilated_mask = cv.dilate(boundary_mask, kernel=np.ones((3, 3), np.uint8), iterations=0)

        bgr_image = cv.bitwise_or(bgr_image, dilated_mask)

        return bgr_image

    def run(self):
        self.init_trackbars()
        while True:
            self.update_trackbars()

            watershedded_image = self.apply_watershed()

            cv.imshow('Watershedded image', watershedded_image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cv.destroyAllWindows()

        gray_water = cv.cvtColor(watershedded_image, cv.COLOR_BGR2GRAY)
        return gray_water

    def process_image(self):
        # self.update_trackbars()

        watershedded_image = self.apply_watershed()

        gray_water = cv.cvtColor(watershedded_image, cv.COLOR_BGR2GRAY)
        return gray_water
