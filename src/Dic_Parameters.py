class Parameters:
    def __init__(self):
        # PATHS
        self.images_path = '.\\datasets\\dice_images'

        # IMAGE PROCESSING
        self.image_processing = {
            # HSV params
            'hue_lower': 0,
            'hue_upper': 179,
            'saturation_lower': 0,
            'saturation_upper': 255,
            'value_lower': 102,
            'value_upper': 255,
            # Image processing operations params
            'threshold_type': 0,  # 0: Binary, 1: Binary Inv, 2: Trunc, 3: ToZero, 4: ToZero Inv
            'threshold_value': 121,
            'gaussian_ksize': 3,
            'median_ksize': 7,
            'dilation_iterations': 1,
            'erosion_iterations': 1,
            'canny_threshold1': 0,
            'canny_threshold2': 400,
            # Contour params
            'retrieval_mode': 1,  # 0: RETR_EXTERNAL, 1: RETR_LIST, 2: RETR_CCOMP, 3: RETR_TREE
            'approximation_method': 1,  # 0: CHAIN_APPROX_NONE, 1: CHAIN_APPROX_SIMPLE
            # Image resolution params
            'width': 820,
            'height': 820,
        }

        self.hough = {
            'dp': 14,  # The inverse ratio of the accumulator resolution to the image resolution (will be divided by 10)
            'minDist': 12,  # Minimum distance between the centers of the detected circles
            'param1': 66,  # Higher threshold for the Canny edge detector (lower is half of this)
            'param2': 12,  # Accumulator threshold for the circle centers at the detection stage
            'minRadius': 7,  # Minimum circle radius to be detected. If unknown, set to 0.
            'maxRadius': 10,  # Maximum circle radius to be detected. If unknown, set to 0.
        }

        self.watershed = {
            'threshold_value': 127,
            'morph_kernel_size': 2,
            'opening_iterations': 0,
            'dilation_iterations': 1,
        }

        self.blobdetection = {
            'minThreshold': 0,
            'maxThreshold': 200,
            'minArea': 20,
            'maxArea': 300,
            'minCircularity': 8,  # Will be divided by 10 in get_detector_params
            'minConvexity': 2,  # Will be divided by 100 in get_detector_params
            'minInertiaRatio': 1  # Will be divided by 100 in get_detector_params
        }
