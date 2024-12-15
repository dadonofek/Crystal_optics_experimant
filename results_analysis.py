import pandas as pd
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def extract_degrees_from_filename(file_name, expression, type=int):
    """
    Extracts a value based on the given expression from a file name.

    :param file_name: The file name string
    :param expression: The regular expression pattern to match
    :param type: The desired type of the extracted value (int or str)
    :return: The extracted value as the specified type, or None if not found
    """
    match = re.search(expression + r'([^\.]+)', file_name)
    if match:
        if type == int:
            return int(match.group(1))  # Extract the number and convert to int
        elif type == str:
            return match.group(1)
    return None


def extract_mean_std_from_csv(file_path):
    """
    Reads a CSV file, extracts data below "Power (W)", and calculates mean and standard error.

    :param file_path: Path to the CSV file
    :return: Mean and standard error of the data
    """
    try:
        # Load the file as a plain text DataFrame
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Find the line index with "Power (W)"
        header_index = None
        for i, line in enumerate(lines):
            if "Power (W)" in line:
                header_index = i
                break

        if header_index is None:
            raise ValueError(f"'Power (W)' header not found in {file_path}")

        # Read the file from the header line onwards
        df = pd.read_csv(file_path, skiprows=header_index)

        # Ensure "Power (W)" is a valid column
        if "Power (W)" not in df.columns:
            raise ValueError(f"Column 'Power (W)' not found in {file_path}")

        # Convert data to numeric, ignoring non-numeric rows
        data = pd.to_numeric(df["Power (W)"], errors='coerce').dropna()

        # Calculate mean and standard error
        mean = data.mean()
        std_err = data.std() / np.sqrt(len(data))
        return mean, std_err

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None


def process_all_csv_in_directory(directory_path, expression=r'_deg', name_type=int, extract_deg=True):
    """
    Processes all CSV files in a directory, extracts mean and standard error,
    and stores them in separate lists, sorted alphabetically by file name.

    :param directory_path: Path to the directory containing CSV files
    :return: Two lists - means and standard errors, sorted by file name
    """
    results = []  # List to store (file_name, mean, std_err)

    # Iterate through files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory_path, file_name)
            mean, std_err = extract_mean_std_from_csv(file_path)
            if mean is not None:
                if extract_deg:
                    deg = extract_degrees_from_filename(file_name, expression, name_type)
                    results.append((deg, mean, std_err))
                else:
                    results.append((file_name, mean, std_err))


    # Sort results by file name
    results.sort(key=lambda x: x[0])
    return results

def proccess_data_B(results):
    # Extract means and standard errors into separate lists
    sorted_deg = [result[0] for result in results]
    sorted_means = [result[1] for result in results]
    sorted_std_errs = [result[2] for result in results]
    return sorted_deg, sorted_means, sorted_std_errs

def proccess_data_D(results):
    # Extract means and standard errors into separate lists
    sorted_deg = np.array([result[0] for result in results])
    sorted_means = np.array([result[1] for result in results])
    sorted_std_errs = np.array([result[2] for result in results])
    return sorted_deg, sorted_means, sorted_std_errs

def ecc_error(I_min, I_max, err_fartor):
    ecc = np.sqrt(1 - (I_min / I_max))
    d_min = 1/(2*I_max*ecc)
    d_max = I_min/(2*I_max**2*ecc)
    return np.sqrt((d_min*I_min*err_fartor)**2 + (d_max*I_max*err_fartor)**2)


def theo_I(E_0, chi, phy, delta):
    chi_rad = np.radians(chi).reshape(-1, 1)  # Reshape chi to (500, 1)
    phy_rad = np.radians(phy)  # No need to reshape phy
    delta_rad = np.radians(delta)

    return E_0 * (
            np.cos(chi_rad) ** 2 -
            np.sin(2 * phy_rad) * np.sin(2 * (phy_rad - chi_rad)) * np.sin(delta_rad / 2) ** 2
    )


def theo_eccentricity(deg):
    I_0 = 1
    chi = np.linspace(0, 360, 500)
    delta = 90  # QWP

    I_values = theo_I(I_0, chi, deg, delta)  # (500, 21) array
    I_min = np.min(I_values, axis=0)  # Min for each deg (shape: 21)
    I_max = np.max(I_values, axis=0)  # Max for each deg (shape: 21)

    return np.sqrt(1 - (I_min / I_max))


def calc_eccentricity(results, err_factor):  # TODO: calc errors
    """
    Calculate eccentricity for each degree value, print min and max intensities, and return lists.

    :param results: List of tuples with degree, max/min values, and their errors
    :return: Two lists - degrees and eccentricities
    """
    eccentricity_data = {}

    # Process results to group by degree
    for item in results:
        degree, intensity, stderr = item  # Unpack degree, intensity, and error
        degree_key, min_or_max = degree.split('_')  # Separate degree and min/max
        if degree_key not in eccentricity_data:
            eccentricity_data[degree_key] = {}
        eccentricity_data[degree_key][min_or_max] = (intensity, stderr)

    eccentricity_data = {int(k): v for k, v in sorted(eccentricity_data.items(), key=lambda item: int(item[0]))}
    # Prepare lists for degrees and eccentricities
    degrees = []
    eccentricities = []
    ecc_err = []
    # Calculate eccentricity
    for degree, values in eccentricity_data.items():
        I_max, I_max_err = values.get("max")
        I_min, I_min_err = values.get("min")
        if I_max is not None and I_min is not None:
            ecc = np.sqrt(1 - (I_min / I_max))
            degrees.append(int(degree))
            eccentricities.append(ecc)
            ecc_err.append(ecc_error(I_min, I_max, err_factor/3))
            # ecc_err.append(ecc_error(I_max, I_max_err, I_min, I_min_err))
            # Print min and max intensities
            print(f"Degree: {degree}°, I_max: {I_max}, I_min: {I_min}, ecc:{ecc}")

    return degrees, eccentricities, ecc_err


def plot_polar(degrees, means, errors):
    # Convert degrees to radians
    angles = np.radians(degrees)

    # Create the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

    # Plot the data points with error bars
    ax.errorbar(angles, means, yerr=errors, fmt='o', color='blue', label="Data Points with Error Bars")

    # Add a trend line (interpolating between points)
    ax.plot(angles, means, color='red', linestyle='-', label="Trend Line")

    # Add labels and formatting
    ax.set_rlabel_position(-22.5)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(r"$\%I$", va='bottom', fontsize=16, color='blue')

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Show plot
    plt.show()


def plot_cartesian(degrees,
                   degree_errors,
                   means,
                   intensity_errors,
                   title,
                   x_label,
                   y_label,
                   type=None,
                   legend_loc='upper right',
                   add_trendline=False,
                   color='blue'):
    # Create the regular Cartesian plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the data points with error bars
    ax.errorbar(degrees, means, xerr=degree_errors, yerr=intensity_errors,
                fmt='.', color='blue', label="Data Points with Error Bars")

    if type == 'HWP':
        degrees_theory = np.linspace(min(degrees), max(degrees), 500)  # Fine grid for smooth curve
        angles_rad_theory = np.radians(degrees_theory)
        I_0 = max(means)  # Normalize initial intensity
        theoretical_intensity = I_0 * np.cos(2*angles_rad_theory - np.pi/2)**2
        ax.plot(degrees_theory, theoretical_intensity, color='red', linestyle='-', label="Theoretical Curve")

    if type == 'QWP':
        # degrees_theory = np.linspace(min(degrees), max(degrees), 500)  # Fine grid for smooth curve
        degrees_theory = degrees
        angles_rad_theory = np.radians(degrees_theory)
        theoretical_ecc = np.array([theo_eccentricity(d) for d in angles_rad_theory])
        ax.plot(degrees_theory, theoretical_ecc, color='red', linestyle='-', label="Theoretical Curve")

    if add_trendline:
        ax.plot(degrees, means, color='red', linestyle='-', label="Trend Line")
    # Add labels and formatting
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=16, color='blue')
    ax.grid(True)
    ax.legend(loc=legend_loc)
    plt.show()


def calculate_extinction_ratios(results):
    # Extract intensities for 2P and 3P
    intensities = {}
    for file, intensity, _ in results:
        key = file.split('_')[0]  # Extract '2P' or '3P'
        if 'max' in file:
            intensities[f"{key}_max"] = intensity
        elif 'min' in file:
            intensities[f"{key}_min"] = intensity

    # Calculate extinction ratios
    extinction_ratio_2P = intensities['2P_max'] / intensities['2P_min']
    extinction_ratio_3P = intensities['3P_max'] / intensities['3P_min']
    print(f'2P_max_intencity = {intensities["2P_max"]}, 2P_min_intencity = {intensities["2P_min"]}, {extinction_ratio_2P = }')
    print(f'3P_max_intencity = {intensities["3P_max"]}, 3P_min_intencity = {intensities["3P_min"]}, {extinction_ratio_3P = }')


def calc_T_R(results):
    # Find the baseline intensity by searching for 'PBS_BL.csv'
    baseline_entry = next(entry for entry in results if entry[0] == 'PBS_BL.csv')
    baseline_intensity = baseline_entry[1]

    # Print the file name and the division result for each entry
    for file_name, intensity, _ in results:
        normalized_intensity = intensity / baseline_intensity
        print(f"File: {file_name}, Normalized Intensity: {normalized_intensity}")


if __name__ == "__main__":
    input_err_factor = 0.036 # from cal_from_0 file, see analyze_cal.py
    # #A- extinction ratio
    # results = process_all_csv_in_directory("/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/A_polaryzers",
    #                                        extract_deg=False)
    # calculate_extinction_ratios(results)
    #
    #
    #
    # #B- HWP
    # results = process_all_csv_in_directory("/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/B_meas")
    # deg, I_mean, I_std_errs = proccess_data_B(results)
    # plot_cartesian(deg, [1] * len(deg), np.array(I_mean)*1000, np.array(I_mean)*input_err_factor*1000,
    #                title="Intensity vs HWP orientation",
    #                x_label="HWP orientation [deg]",
    #                y_label="Intensity [mV]",
    #                type='HWP')
    #
    # # C- QWP
    # results = process_all_csv_in_directory(directory_path="/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/C_meas",
    #                                        name_type=str)
    # deg, ecc, ecc_err = calc_eccentricity(results, input_err_factor)
    #
    # plot_cartesian(deg, [1] * len(deg), ecc, ecc_err,
    #                title="eccentricity vs QWP orientation",
    #                x_label="QWP orientation [deg]",
    #                y_label="eccentricity",
    #                type='QWP',
    #                legend_loc='lower right')


    # # D- opto isolator
    # results = process_all_csv_in_directory("/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/PBS_new",
    #                                        extract_deg=False)
    # calc_T_R(results)


    R_results = process_all_csv_in_directory(
        directory_path="/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/D_meas_R",
        expression=r'iso_r')
    T_results = process_all_csv_in_directory(
        directory_path="/Users/dadonofek/PycharmProjects/pythonProject/flask/lab_5/D_meas_T",
        expression=r'iso_t')
    R_deg, R_I_mean, R_I_std_errs = proccess_data_D(R_results)
    T_deg, T_I_mean, T_I_std_errs = proccess_data_D(T_results)
    # plot_cartesian(R_deg, [2] * len(R_deg), R_I_mean*1000, R_I_mean*input_err_factor*1000,
    #                title="R_Opto-isolator",
    #                x_label="QWP orientation [deg]",
    #                y_label="Intensity [mW]",
    #                add_trendline=True)
    # plot_cartesian(T_deg, [2] * len(T_deg), T_I_mean * 1000, T_I_mean * input_err_factor * 1000,
    #                title="T_Opto-isolator",
    #                x_label="QWP orientation [deg]",
    #                y_label="Intensity [mW]",
    #                add_trendline=True)

    # Calculate the sum of R_I_mean and T_I_mean
    RT_I_mean_sum = R_I_mean + T_I_mean
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.errorbar(R_deg, R_I_mean * 1000,xerr=[1] * len(R_deg), yerr=R_I_mean * input_err_factor * 1000, fmt='.', label="Reflection", color='blue')
    ax.plot(R_deg, R_I_mean * 1000, color='blue', linestyle='-')

    ax.errorbar(T_deg, T_I_mean * 1000,xerr=[1] * len(T_deg), yerr=T_I_mean * input_err_factor * 1000, fmt='.', label="Transmission", color='green')
    ax.plot(T_deg, T_I_mean * 1000, color='green', linestyle='-')

    ax.plot(R_deg, RT_I_mean_sum * 1000, label="Sum (R + T)", color='red', linestyle='--')

    ax.set_title("Reflection and Transmission of Opto-isolator", fontsize=16, color='blue')
    ax.set_xlabel("QWP orientation [deg]", fontsize=12)
    ax.set_ylabel("Intensity [mW]", fontsize=12)
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.show()



