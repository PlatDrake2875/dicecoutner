import cv2 as cv
import numpy as np


class BlobDetectorTrackbar:
    def __init__(self, window_name, image, params):
        self.window_name = window_name
        self.image = image

        self.params = params
        self.init_trackbars()

    def nothing(self, x):
        pass

    def init_trackbars(self):
        cv.namedWindow(self.window_name)
        cv.resizeWindow(self.window_name, 400, 400)
        # Max values for each parameter
        max_values = {
            'minThreshold': 255,
            'maxThreshold': 255,
            'minArea': 500,
            'maxArea': 1000,
            'minCircularity': 10,
            'minConvexity': 100,
            'minInertiaRatio': 100
        }
        for param_name, initial in self.params.items():
            cv.createTrackbar(param_name, self.window_name, initial, max_values[param_name], self.nothing)

    def update_trackbars(self):
        # Define valid ranges for each parameter
        valid_ranges = {
            'minThreshold': (0, 255),
            'maxThreshold': (0, 255),
            'minArea': (1, 5000),
            'maxArea': (1, 10000),  # Ensure this range is appropriate for your application
            'minCircularity': (1, 10),  # Actual range is 0.0-1.0, but scaled by 10
            'minConvexity': (1, 100),  # Actual range is 0.0-1.0, but scaled by 100
            'minInertiaRatio': (1, 100)  # Actual range is 0.0-1.0, but scaled by 100
        }

        for key in self.params.keys():
            trackbar_val = cv.getTrackbarPos(key, self.window_name)
            min_val, max_val = valid_ranges[key]

            # Validate the trackbar value
            if trackbar_val < min_val:
                print(f"Warning: {key} value {trackbar_val} is below the minimum {min_val}. Correcting to {min_val}.")
                trackbar_val = min_val
            elif trackbar_val > max_val:
                print(f"Warning: {key} value {trackbar_val} is above the maximum {max_val}. Correcting to {max_val}.")
                trackbar_val = max_val

            self.params[key] = trackbar_val

    def get_detector_params(self):
        detector_params = cv.SimpleBlobDetector_Params()
        self.update_trackbars()

        detector_params.minThreshold = self.params['minThreshold']
        detector_params.maxThreshold = self.params['maxThreshold']
        detector_params.filterByArea = True
        detector_params.minArea = self.params['minArea']
        detector_params.maxArea = self.params['maxArea']
        detector_params.filterByCircularity = True
        detector_params.minCircularity = self.params['minCircularity'] / 10.0
        detector_params.filterByConvexity = True
        detector_params.minConvexity = self.params['minConvexity'] / 100.0
        detector_params.filterByInertia = True
        detector_params.minInertiaRatio = self.params['minInertiaRatio'] / 100.0
        return detector_params

    def detect_and_display_blobs(self):
        detector_params = self.get_detector_params()
        detector = cv.SimpleBlobDetector_create(detector_params)
        keypoints = detector.detect(self.image)
        im_with_keypoints = cv.drawKeypoints(self.image, keypoints, np.array([]), (0, 0, 255),
                                             cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow('Blobbed image', im_with_keypoints)

    def run(self):
        while True:
            self.detect_and_display_blobs()
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cv.destroyAllWindows()

    def count_circles(self):
        detector_params = self.get_detector_params()
        detector = cv.SimpleBlobDetector_create(detector_params)
        keypoints = detector.detect(self.image)

        cv.destroyAllWindows()
        return len(keypoints)
