import glob
import cv2 as cv
import numpy as np

import Utilities as utils
from src.trackbars.ImageProcessingTrackbar import ImageProcessingTrackbar
from src.trackbars.SimpleBlobDetectorTrackbar import BlobDetectorTrackbar
from src.trackbars.WatershedTrackbar import WatershedTrackbar


class DicePointCounter:
    def __init__(self, image, image_params, watershed_params, blob_params):
        self.image = image
        self.image_params = image_params
        self.watershed_params = watershed_params
        self.blob_params = blob_params

    def process_dice(self, title='Image'):
        original_image = self.image.copy()
        image_trackbar = ImageProcessingTrackbar(title='Image slider', image=self.image, params=self.image_params)
        image = image_trackbar.process_image()

        watershed_trackbar = WatershedTrackbar(title='Watershed Slider', image=image, params=self.watershed_params)
        image = watershed_trackbar.process_image()

        sbd_trackbar = BlobDetectorTrackbar(window_name='Blob Slider', image=image, params=self.blob_params)
        cnt_pimples = sbd_trackbar.count_circles()

        print(f"Detected {cnt_pimples} points!")

        utils.display_image(title, original_image)