import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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

def show_image_comp(theo_data, result_data):
    # Display images side by side
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].imshow(theo_data, cmap='gray')
    axes[0].plot([0, theo_data.shape[1] - 1], [theo_data.shape[0] - 1, 0], color="red", linestyle="--")
    axes[0].set_title("Theoretical Image")
    axes[0].axis("off")

    axes[1].imshow(result_data)
    axes[1].plot([0, theo_data.shape[1] - 1], [theo_data.shape[0] - 1, 0], color="red", linestyle="--")
    axes[1].set_title("Result Image")
    axes[1].axis("off")

    plt.show()


def plot_all(paths, colors, shift=0, plot_indices=None, filter_win=1):
    """
    Plot theoretical and result light intensities in separate plots for multiple colors.

    Args:
        paths (list of tuples): List of (result_path, theo_path) for each color.
        colors (list of str): List of color labels.
        shift (int): Shift to apply to result intensity.
        plot_indices (list or tuple): [start, end] range to plot specific indices.
        filter_win (int): Window size for filtering intensities.
    """
    # Initialize lists to store data for both plots
    all_theo_intensities = []
    all_result_intensities = []

    for (result_path, theo_path), color in zip(paths, colors):
        # Load images
        theo_image = Image.open(theo_path).convert("L")
        result_image = Image.open(result_path).convert("L")
        theo_data = np.array(theo_image)
        result_data = np.array(result_image)

        # Extract diagonal intensities
        theo_intensity = np.diagonal(np.flipud(theo_data))
        result_intensity = np.diagonal(np.flipud(result_data))

        # Apply shift to result intensity
        if shift != 0:
            result_intensity = np.roll(result_intensity, shift)

        # Apply smoothing filter
        result_intensity = np.convolve(result_intensity, np.ones(filter_win) / filter_win, mode='same')

        # Normalize data
        result_intensity = result_intensity * (theo_intensity.max() / result_intensity.max())
        min_v = result_intensity[(result_intensity.shape[0] // 2 - 50):(result_intensity.shape[0] // 2 + 50)].min()
        max_v = result_intensity.max()
        result_intensity = (result_intensity - min_v) * (max_v / (max_v - min_v))
        result_intensity /= 255
        theo_intensity = theo_intensity.copy() / 255

        # Apply plot_indices
        if plot_indices:
            start, end = plot_indices
            theo_intensity = theo_intensity[start:end]
            result_intensity = result_intensity[start:end]

        # Store intensities for separate plots
        all_theo_intensities.append((theo_intensity, color))
        all_result_intensities.append((result_intensity, color))

    # Plot theoretical intensities
    plt.figure(figsize=(12, 6))
    for (intensity, color), plot_color in zip(all_theo_intensities, ['purple', 'green', 'red']):
        plt.plot(intensity, label=f"Theoretical Intensity ({color})", linestyle="-", color=plot_color)
    plt.xlabel("Point along diagonal [pixels]")
    plt.ylabel("Intensity [arbitrary units]")
    plt.ylim(0, 1.1)
    plt.title("Theoretical Intensity Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot result intensities
    plt.figure(figsize=(12, 6))
    for (intensity, color), plot_color in zip(all_result_intensities, ['purple', 'green', 'red']):
        plt.plot(intensity, label=f"Result Intensity ({color})", color=plot_color)
    plt.xlabel("Point along diagonal [pixels]")
    plt.ylabel("Intensity [arbitrary units]")
    plt.ylim(0, 1.1)
    plt.title("Result Intensity Comparison")
    plt.legend()
    plt.grid()
    plt.show()

def plot_intensity_theo_comp(theo_path,
                             result_path,
                             color,
                             shift=0,
                             plot_indices=None,
                             show_images=True,
                             filter_win=1):
    """
    Plot theoretical and result light intensities along the diagonal axis of images,
    and display both images side by side.

    Args:
        theo_path (str): Path to the theoretical image file.
        result_path (str): Path to the result image file.
        shift (int): Number of indices to shift the result intensity.
        plot_indices (list or tuple): [start, end] range to plot specific indices.
    """
    # Load theoretical and result images
    theo_image = Image.open(theo_path).convert("L")
    result_image = Image.open(result_path).convert("L")
    theo_data = np.array(theo_image)
    result_data = np.array(result_image)

    if show_images:
        show_image_comp(theo_data, Image.open(result_path))

    # Extract diagonal intensities
    theo_intensity = np.diagonal(np.flipud(theo_data))
    result_intensity = np.diagonal(np.flipud(result_data))

    # Apply the shift to result intensity
    if shift != 0:
        result_intensity = np.roll(result_intensity, shift)

    result_intensity = np.convolve(result_intensity, np.ones(filter_win) / filter_win, mode='same')

    # Normalize data
    result_intensity = result_intensity * (theo_intensity.max()/ result_intensity.max())
    min_v = result_intensity[(result_intensity.shape[0] // 2 - 50):(result_intensity.shape[0] // 2 + 50)].min()
    max_v = result_intensity.max()
    result_intensity = (result_intensity - min_v) * (max_v / (max_v - min_v))
    result_intensity /= 255
    theo_intensity = theo_intensity.copy() / 255


    # Apply plot_indices to select the range for plotting
    if plot_indices:
        start, end = plot_indices
        theo_intensity = theo_intensity[start:end]
        result_intensity = result_intensity[start:end]

    # # Plot theoretical and result intensities
    # np.savetxt('/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/simulation/test/green_arrays/theo_intensity.csv',
    #         theo_intensity, delimiter=',', fmt='%f')
    # np.savetxt(
    #     '/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/simulation/test/green_arrays/result_intensity.csv',
    #     result_intensity, delimiter=',', fmt='%f')
    plt.figure(figsize=(10, 6))
    plt.plot(theo_intensity, label="Theoretical Intensity", color="red", linestyle="--")
    plt.plot(result_intensity, label="Result Intensity", color="blue")
    plt.xlabel("Point along diagonal [pixels]")
    plt.ylabel("Intensity [arbitrary units]")
    plt.ylim(0, 1.1)
    plt.title(f"Conoscopy therethical comparison for {color} lihgt")
    plt.legend()
    plt.grid()
    plt.show()


def calc_pixel_length():
    l = np.sqrt(9 ** 2 + 0.3 ** 2)
    green_np = np.sqrt(555 ** 2 + 86 ** 2)
    green_pl = l / green_np

    red_np = np.sqrt(707 ** 2 + 64 ** 2)
    red_pl = l / red_np

    purple_np = np.sqrt(491 ** 2 + 70 ** 2)
    purple_pl = l / purple_np

    white_np = np.sqrt(642 ** 2 + 53 ** 2)
    white_pl = l / white_np
    # print(f'{white_pl = }\n{red_pl = }\n{purple_pl = }\n{green_pl = }\n')
    return white_pl, purple_pl, red_pl, green_pl


if __name__ == "__main__":
    white_pl, purple_pl, red_pl, green_pl = calc_pixel_length()

    # TEST
    purple_path = '/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/reformated/purple_filtered.jpg'
    purple_sim_path = "/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/simulation/sim_405_circle.png"
    green_path = '/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/reformated/green_filtered.jpg'
    green_sim_path = "/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/simulation/sim_532_circle.png"
    red_path = '/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/reformated/red_filtered.jpg'
    red_sim_path = "/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/conoscopic_images/simulation/sim_628_circle.png"

    # plot_light_intensity_filtered(purple_sim_path, show_image=True)
    # plot_light_intensity_filtered(purple_path, show_image=True)
    plot_intensity_theo_comp(result_path=red_path,
                             theo_path=red_sim_path,
                             color="532 nm",
                             shift=0,
                             plot_indices=[70, 471],
                             show_images=True,
                             filter_win=15)


    # # plot all on the same graph
    # paths = [
    #     (purple_path, purple_sim_path),
    #     (green_path, green_sim_path),
    #     (red_path, red_sim_path)
    # ]
    # colors = ["405 nm", "532 nm", "628 nm"]
    # plot_all(paths, colors, shift=0, plot_indices=[70, 471], filter_win=15)



    # # play with green
    #
    # color = "532 nm"
    # filter_win = 8
    # shift = 0
    # result_intensity = np.loadtxt('/conoscopic_images/simulation/test/green_arrays/result_intensity.csv', delimiter=',')
    # theo_intensity = np.loadtxt('/conoscopic_images/simulation/test/green_arrays/theo_intensity.csv', delimiter=',')
    # result_intensity = np.roll(result_intensity, shift)
    # result_intensity = np.convolve(result_intensity, np.ones(filter_win) / filter_win, mode='same')
    # # Plot theoretical and result intensities
    # plt.figure(figsize=(10, 6))
    # plt.plot(theo_intensity, label="Theoretical Intensity", color="red", linestyle="--")
    # plt.plot(result_intensity, label="Result Intensity", color="blue")
    # plt.xlabel("Point along diagonal [pixels]")
    # plt.ylabel("Intensity [arbitrary units]")
    # plt.ylim(0, 1.1)
    # plt.title(f"Conoscopy therethical comparison for {color} lihgt")
    # plt.legend()
    # plt.grid()
    # plt.show()
    #


