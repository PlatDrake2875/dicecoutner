import cv2 as cv
import numpy as np


def nothing(x):
    pass


class ImageProcessingTrackbar:
    def __init__(self, title, image, params):
        self.original_image = image
        self.image = image
        self.title = title
        self.params = params

    def init_trackbars(self):
        cv.namedWindow(self.title)

        cv.createTrackbar('Hue Lower', self.title, self.params['hue_lower'], 179, nothing)
        cv.createTrackbar('Hue Upper', self.title, self.params['hue_upper'], 179, nothing)
        cv.createTrackbar('Saturation Lower', self.title, self.params['saturation_lower'], 255, nothing)
        cv.createTrackbar('Saturation Upper', self.title, self.params['saturation_upper'], 255, nothing)
        cv.createTrackbar('Value Lower', self.title, self.params['value_lower'], 255, nothing)
        cv.createTrackbar('Value Upper', self.title, self.params['value_upper'], 255, nothing)
        cv.createTrackbar('Threshold Type', self.title, self.params['threshold_type'], 4, nothing)
        cv.createTrackbar('Threshold Value', self.title, self.params['threshold_value'], 255, nothing)
        cv.createTrackbar('Gaussian Ksize', self.title, self.params['gaussian_ksize'], 11, nothing)
        cv.createTrackbar('Median Ksize', self.title, self.params['median_ksize'], 11, nothing)
        cv.createTrackbar('Dilation Iterations', self.title, self.params['dilation_iterations'], 10, nothing)
        cv.createTrackbar('Erosion Iterations', self.title, self.params['erosion_iterations'], 10, nothing)
        cv.createTrackbar('Canny Threshold1', self.title, self.params['canny_threshold1'], 1000, nothing)
        cv.createTrackbar('Canny Threshold2', self.title, self.params['canny_threshold2'], 1000, nothing)
        cv.createTrackbar('Retrieval Mode', self.title, self.params['retrieval_mode'], 3, nothing)
        cv.createTrackbar('Approximation Method', self.title, self.params['approximation_method'], 1, nothing)
        cv.createTrackbar('Width', self.title, self.params['width'], 1920, nothing)
        cv.createTrackbar('Height', self.title, self.params['height'], 1080, nothing)

    def update_trackbars(self):
        for key in self.params.keys():
            self.params[key] = cv.getTrackbarPos(key.replace('_', ' ').title(), self.title)

    def apply_hsv_filter(self, image):
        if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale or single channel
            hsv_image = cv.cvtColor(cv.cvtColor(image, cv.COLOR_GRAY2BGR), cv.COLOR_BGR2HSV)
        else:
            hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        lower_bound = np.array([self.params['hue_lower'], self.params['saturation_lower'], self.params['value_lower']])
        upper_bound = np.array([self.params['hue_upper'], self.params['saturation_upper'], self.params['value_upper']])
        mask = cv.inRange(hsv_image, lower_bound, upper_bound)

        if len(image.shape) == 2 or image.shape[2] == 1:  # Image is grayscale
            image_bgr = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            filtered_image = cv.bitwise_and(image_bgr, image_bgr, mask=mask)
        else:
            filtered_image = cv.bitwise_and(image, image, mask=mask)

        return filtered_image

    def apply_threshold(self, image):
        threshold_types = [cv.THRESH_BINARY, cv.THRESH_BINARY_INV, cv.THRESH_TRUNC, cv.THRESH_TOZERO,
                           cv.THRESH_TOZERO_INV]
        _, thresh_image = cv.threshold(image, self.params['threshold_value'], 255,
                                       threshold_types[self.params['threshold_type']])
        return thresh_image

    def apply_gaussian_blur(self, image):
        ksize = max(1, self.params['gaussian_ksize'] // 2 * 2 + 1)  # Ensure kernel size is odd
        blurred_image = cv.GaussianBlur(image, (ksize, ksize), 0)
        return blurred_image

    def apply_median_blur(self, image):
        ksize = max(1, self.params['median_ksize'] // 2 * 2 + 1)  # Ensure kernel size is odd
        blurred_image = cv.medianBlur(image, ksize)
        return blurred_image

    def apply_dilation(self, image):
        kernel = np.ones((5, 5), np.uint8)
        dilated_image = cv.dilate(image, kernel, iterations=self.params['dilation_iterations'])
        return dilated_image

    def apply_erosion(self, image):
        kernel = np.ones((5, 5), np.uint8)
        eroded_image = cv.erode(image, kernel, iterations=self.params['erosion_iterations'])
        return eroded_image

    def apply_canny(self, image):
        canny_image = cv.Canny(image, self.params['canny_threshold1'], self.params['canny_threshold2'], L2gradient=True)
        return canny_image

    @staticmethod
    def apply_find_contours(image, retrieval_mode, approximation_method):
        mode_map = [cv.RETR_EXTERNAL, cv.RETR_LIST, cv.RETR_CCOMP, cv.RETR_TREE]
        method_map = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE]
        contours, hierarchy = cv.findContours(image, mode_map[retrieval_mode], method_map[approximation_method])
        return contours, hierarchy

    def draw_contours_on_image(self, image):
        # Adjust to use the new dynamic parameters
        contours, hierarchy = self.apply_find_contours(
            image,
            self.params['retrieval_mode'],
            self.params['approximation_method']
        )

        contour_image = image.copy()
        # cv.drawContours(contour_image, contours, -1, (255, 255, 255), 2, lineType=cv.LINE_AA)
        contour_image = self.draw_contours_with_blurring(contour_image, contours, (255, 255, 255))
        return contour_image

    @staticmethod
    def draw_contours_with_custom_thickness(contour_image, contours, contour_color):
        # Scale the image up
        scale_factor = 1  # Experiment with this value
        width = int(contour_image.shape[1] * scale_factor)
        height = int(contour_image.shape[0] * scale_factor)
        dim = (width, height)
        resized_image = cv.resize(contour_image, dim, interpolation=cv.INTER_CUBIC)

        cv.drawContours(resized_image, contours, -1, contour_color, scale_factor, lineType=cv.LINE_8)

        original_dim = (contour_image.shape[1], contour_image.shape[0])
        final_image = cv.resize(resized_image, original_dim, interpolation=cv.INTER_LINEAR)

        return final_image

    @staticmethod
    def draw_contours_with_blurring(contour_image, contours, contour_color):
        # Create a blank image with the same dimensions as the contour image
        blank_image = np.zeros_like(contour_image)

        # Draw contours with a thickness of 1
        cv.drawContours(blank_image, contours, -1, contour_color, 8, lineType=cv.LINE_AA)

        # Apply a slight Gaussian blur to the contours
        blurred_contours = cv.GaussianBlur(blank_image, (3, 3), 0)

        # Add the blurred contours to the original image
        final_image = cv.addWeighted(contour_image, 1, blurred_contours, 0.15, 0)

        return final_image

    def apply_resize(self, image):
        new_width = self.params['width']
        new_height = self.params['height']

        if new_width < self.original_image.shape[1] or new_height < self.original_image.shape[0]:
            print("Resolution smaller than original!")
            return image

        image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
        return image

    def run(self):
        self.init_trackbars()
        while True:
            self.update_trackbars()

            processed_image = self.apply_hsv_filter(self.image)
            processed_image = self.apply_resize(processed_image)

            processed_image = self.apply_median_blur(processed_image)
            processed_image = self.apply_gaussian_blur(processed_image)

            processed_image = self.apply_dilation(processed_image)
            processed_image = self.apply_erosion(processed_image)

            processed_image = self.apply_canny(processed_image)

            processed_image = self.draw_contours_on_image(processed_image)

            cv.imshow('Processed image', processed_image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cv.destroyWindow('Processed image')
        cv.destroyWindow(self.title)
        # cv.waitKey(1)

        return processed_image

    def process_image(self):
        # self.update_trackbars()
        processed_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image

        processed_image = self.apply_hsv_filter(processed_image)
        processed_image = self.apply_resize(processed_image)

        processed_image = self.apply_median_blur(processed_image)
        processed_image = self.apply_gaussian_blur(processed_image)

        processed_image = self.apply_dilation(processed_image)
        processed_image = self.apply_erosion(processed_image)

        processed_image = self.apply_canny(processed_image)

        processed_image = self.draw_contours_on_image(processed_image)

        return processed_image
