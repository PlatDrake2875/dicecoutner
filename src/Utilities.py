import glob
import numpy as np
import cv2 as cv
import os


def get_images(path, percentage=1.0):
    """
    Method that returns the first n% of images
    :param percentage: of images (1.0 by default)
    :return: First n% of images
    """

    pattern = os.path.join(path, "*.jpg")
    images_paths = glob.glob(pattern)
    print(f"Found {len(images_paths)} image paths")

    cnt_images = int(len(images_paths) * percentage)
    images = [cv.imread(images_paths[i], cv.IMREAD_GRAYSCALE) for i in range(cnt_images)]
    print(f"Read the first {percentage * 100}% images.")
    print(f"Which ammounts to {len(images)}")

    return images


def draw_circles(image, circles):
    """
    Actually draws circles on an image.
    :param image:
    :param circles: tuple (x,y,radius)
    :return:
    """
    if circles is None:
        print("No circles to draw!")
        return

    # print(f"Trying to draw {len(circles[0, :])} circles")

    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for circle_data in circles[0, :]:
        circle_data = np.uint16(np.around(circle_data))
        center = (circle_data[0], circle_data[1])
        radius = circle_data[2]
        # print(f"Drawing circle at coords {center} and radius {radius}")
        cv.circle(image,
                  center=center,
                  radius=radius,
                  color=(0, 255, 0),  # Outline color
                  thickness=1)
    return image


def display_circles(title, image, circles):
    """
    Draws circles on image
    :param title:
    :param image:
    :param circles:
    :return:
    """
    if circles is None:
        print("No circles to draw!")
        return

    draw_image = draw_circles(image, circles)

    # draw_image = cv.resize(draw_image, dsize=(0, 0), fx=2, fy=2)
    cv.imshow(title, draw_image)


def display_image(title, image):
    """
    Displays image on screen. Closes it by pressing 'q'.
    :param title:
    :param image:
    :return:
    """
    cv.imshow(title, image)

    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv.destroyAllWindows()
