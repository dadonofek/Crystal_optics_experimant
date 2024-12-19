import numpy as np

def jones_qwp_output(input_state, theta=0):
    """
    Calculate the Jones output state of polarized light after passing through a QWP.

    Parameters:
        input_state (numpy.ndarray): Input Jones vector (2x1 array).
        theta (float): Angle of the fast axis of the QWP (in degrees) with respect to the x-axis.

    Returns:
        numpy.ndarray: Output Jones vector (2x1 array).
    """
    # Convert theta to radians
    theta_rad = np.deg2rad(theta)

    # Define the QWP Jones matrix
    qwp_matrix = np.array([
        [np.cos(theta_rad)**2 + 1j * np.sin(theta_rad)**2, (1 - 1j) * np.sin(theta_rad) * np.cos(theta_rad)],
        [(1 - 1j) * np.sin(theta_rad) * np.cos(theta_rad), np.sin(theta_rad)**2 + 1j * np.cos(theta_rad)**2]
    ])

    # Calculate the output state
    output_state = np.dot(qwp_matrix, input_state)

    return output_state

def jones_intensity(jones_vector):
    """
    Calculate the total intensity of a Jones vector.

    Parameters:
        jones_vector (numpy.ndarray): Jones vector (2x1 array).

    Returns:
        float: Total intensity of the light.
    """
    return np.abs(jones_vector[0, 0])**2 + np.abs(jones_vector[1, 0])**2

def jones_polarizer(input_state, theta=0):
    """
    Calculate the Jones output state of polarized light after passing through a polarizer.

    Parameters:
        input_state (numpy.ndarray): Input Jones vector (2x1 array).
        theta (float): Angle of the polarizer's transmission axis (in degrees) with respect to the x-axis.

    Returns:
        numpy.ndarray: Output Jones vector (2x1 array).
    """
    # Convert theta to radians
    theta_rad = np.deg2rad(theta)

    # Define the polarizer Jones matrix
    polarizer_matrix = np.array([
        [np.cos(theta_rad)**2, np.cos(theta_rad) * np.sin(theta_rad)],
        [np.cos(theta_rad) * np.sin(theta_rad), np.sin(theta_rad)**2]
    ])

    # Calculate the output state
    output_state = np.dot(polarizer_matrix, input_state)

    return output_state

# Example usage:
if __name__ == "__main__":
    # Define an input Jones vector (e.g., horizontal polarization)
    input_state = np.array([[1], [0]])

    # QWP orientation (45 degrees)
    theta_qwp = 5

    # Calculate the output state after QWP
    output_state_qwp = jones_qwp_output(input_state, theta_qwp)

    print("Output Jones vector after QWP:")
    print(output_state_qwp)

    # Calculate and print the total intensity after QWP
    intensity_qwp = jones_intensity(output_state_qwp)
    print("Total intensity after QWP:", intensity_qwp)

    # Polarizer orientation (30 degrees)
    theta_polarizer = 30

    # Calculate the output state after the polarizer
    output_state_polarizer = jones_polarizer(output_state_qwp, theta_polarizer)

    print("Output Jones vector after polarizer:")
    print(output_state_polarizer)

    # Calculate and print the total intensity after the polarizer
    intensity_polarizer = jones_intensity(output_state_polarizer)
    print("Total intensity after polarizer:", intensity_polarizer)
