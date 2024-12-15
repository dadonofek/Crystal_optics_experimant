import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def conoscopic_pattern(size=500, wavelength=550e-9, thickness=1e-3, ne=1.486, no=1.658, max_angle=10):
    # Create a grid of angles
    lim = np.tan(max_angle)
    x = np.linspace(-lim, lim, size)
    y = np.linspace(-lim, lim, size)
    X, Y = np.meshgrid(x, y)

    # Calculate the angle from the optical axis
    theta = np.arctan(np.sqrt(X ** 2 + Y ** 2))

    # Calculate the phase difference using equation 9
    delta = (2 * np.pi / wavelength) * thickness * (ne - no) * np.sin(theta) ** 2

    # Calculate the angle between polarizer and crystal axis
    phi = np.arctan2(Y, X)

    # Calculate intensity using equation 9
    intensity = np.sin(2 * phi) ** 2 * np.sin(delta / 2) ** 2

    return intensity

def conoscopic_pattern_circle(size=500, wavelength=550e-9, thickness=1e-3, ne=1.486, no=1.658, max_angle=10):
    # Create a grid of angles
    lim = np.tan(max_angle)
    x = np.linspace(-lim, lim, size)
    y = np.linspace(-lim, lim, size)
    X, Y = np.meshgrid(x, y)

    # Calculate the angle from the optical axis
    theta = np.arctan(np.sqrt(X ** 2 + Y ** 2))

    # Calculate the phase difference using equation 9
    delta = (2 * np.pi / wavelength) * thickness * (ne - no) * np.sin(theta) ** 2

    # Calculate the angle between polarizer and crystal axis
    phi = np.arctan2(Y, X)

    # Calculate intensity using equation 9
    intensity = np.sin(2 * phi) ** 2 * np.sin(delta / 2) ** 2

    # Black out pixels outside the main circle
    radius = size // 2
    center_x, center_y = size // 2, size // 2
    Y_idx, X_idx = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X_idx - center_x) ** 2 + (Y_idx - center_y) ** 2)
    mask = dist_from_center <= radius
    intensity[~mask] = 0  # Black out regions outside the circle

    return intensity



def show_image_with_line(image_path):
    """
    Display the image and overlay the diagonal line (bottom-left to top-right).

    Args:
        image_path (str): Path to the image file.
    """
    # Load the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    data = np.array(image)

    # Plot the image with the diagonal line
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap="gray")
    plt.plot([0, data.shape[1] - 1], [data.shape[0] - 1, 0], color="red", linestyle="--", label="Diagonal")
    plt.title("Image with Diagonal Line")
    plt.legend()
    plt.axis("off")
    plt.show()

    return data


def plot_light_intensity_filtered(image_path, show_image=False):
    """
    Plot the light intensity along the specified diagonal axis of an image,
    excluding values with intensity <= 40.
    """
    if show_image:
        show_image_with_line(image_path)
    # Load the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    data = np.array(image)
    # Extract diagonal points from bottom-left to top-right
    diagonal_intensity = np.diagonal(np.flipud(data))

    # Filter to remove tails with intensity <= 40
    filtered_indices = np.where(diagonal_intensity > 40)[0]
    filtered_intensity = diagonal_intensity[filtered_indices]

    # Plot the intensity along the diagonal
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_indices, filtered_intensity, label="Filtered Intensity along diagonal")
    plt.xlabel("Point along diagonal (pixels)")
    plt.ylabel("Intensity")
    plt.title("Light Intensity Along Diagonal (Filtered)")
    plt.legend()
    plt.grid()
    plt.show()

def plot(image, save=False, save_to=None):
    plt.figure(figsize=(7, 7))
    plt.imshow(image, cmap='gray')
    # plt.title('Simulated Conoscopic Pattern for Uniaxial Crystal')
    plt.axis('off')
    if save:
        plt.savefig(save_to, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    # gen simulation
    l = 11.6
    h = 4.5
    max_angle = np.atan(h / l)/2
    # [405e-9, 532e-9, 628e-9]
    wavelengths = [405e-9, 532e-9, 628e-9]
    for wl in wavelengths:
        pattern = conoscopic_pattern_circle(size=550,
                                            wavelength=wl,
                                            thickness=1.1e-3,
                                            ne=2.2116,
                                            no=2.3007,
                                            max_angle=max_angle)
        plot(pattern,
             save=True,
             save_to=f"/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images"
                     f"/simulation/sim_{str(int(wl * 10 ** 9))}_circle.png")


