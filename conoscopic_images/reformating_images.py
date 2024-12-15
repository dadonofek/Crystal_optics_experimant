import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image


def filter_color(path, lower_bounds, upper_bounds, save_to):
    image = Image.open(path)
    new_image_array = np.array(image)

    # Convert to HSV color space for better color segmentation
    hsv_new_image = cv2.cvtColor(new_image_array, cv2.COLOR_RGB2HSV)

    # Create masks for color
    combined_mask = None
    for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
        mask = cv2.inRange(hsv_new_image, lower_bound, upper_bound)
        combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

    result = cv2.bitwise_and(new_image_array, new_image_array, mask=combined_mask)

    # Plot the original and the processed image
    plt.figure(figsize=(7, 7))
    plt.imshow(result)
    plt.axis("off")
    if save_to is not None:
        plt.savefig(save_to, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    image_path = '/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/reformated/red.jpg'
    save_to = '/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/reformated/red_filtered.jpg'

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Pass both red ranges
    filter_color(
        path=image_path,
        lower_bounds=[lower_red1, lower_red2],
        upper_bounds=[upper_red1, upper_red2],
        save_to=save_to
    )

    # # rotate:
    # image_path = "/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/reformated/green.jpg"
    # image = Image.open(image_path)
    # rotated = image.rotate(1, expand=True)
    # rotated.save('/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/reformated/green.jpg')
