import cv2 as cv

from src.DicePointCounter import DicePointCounter
from src.Dic_Parameters import Parameters
import Utilities as utils

params = Parameters()


def dice_point_counter():
    images = utils.get_images(params.images_path, percentage=0.1)

    for idx, image in enumerate(images):
        print("==============================")
        print(f"Querying image {idx + 1}")

        solver = DicePointCounter(image=image,
                                  image_params=params.image_processing,
                                  watershed_params=params.watershed,
                                  blob_params=params.blobdetection)

        solver.process_dice(f"Image {idx + 1}")


def main():
    dice_point_counter()


if __name__ == "__main__":
    main()
