import numpy as np
import matplotlib.pyplot as plt

# Define constants
I_0 = 1
phy_angles = [0, 20, 45]
chi = np.linspace(0, 360, 1000)
delta = 90 # QWP

def I(E_0, chi, phy, delta):
    chi_rad = np.radians(chi)
    phy_rad = np.radians(phy)
    delta_rad = np.radians(delta)
    # print(f"{chi_rad=}, {phy_rad=}, {delta_rad=}")
    return E_0 * (np.cos(chi_rad)**2 -
                  np.sin(2 * phy_rad)
                   * np.sin(2 * (phy_rad - chi_rad))
                    * np.sin(delta_rad / 2)**2)

# Plotting
plt.figure(figsize=(10, 6))
for phy in phy_angles:
    # Calculate intensity as a function of theta for each alpha
    intensity = I(I_0, chi, phy, delta)
    plt.plot(chi, intensity, label=f"Polarizer-QWP Angle = {phy}°")

# Labels and legend
plt.xlabel("Analyzer Angle (degrees)")
plt.ylabel("Normalized Intensity")
plt.title("Intensity vs Analyzer Angle for Various Polarizer-QWP Angles")
plt.legend()
plt.grid(True)
# plt.show()

def eccentricity(theta):
  chi = np.linspace(0, 360, 1000)
  delta = 90 # QWP
  I_min = min(I(I_0, chi, theta, delta))
  I_max = max(I(I_0, chi, theta, delta))
  # print(f"{I_min=}, {I_max=}")
  return np.sqrt(1 - (I_min/I_max))

theta = np.linspace(0, 180, 181)
ecc = [eccentricity(t) for t in theta]

plt.figure(figsize=(10, 6))
plt.plot(theta, ecc)
plt.xlabel("Theta (degrees)")
plt.ylabel("Eccentricity")
plt.title("Eccentricity vs Theta")
plt.grid(True)
plt.show()

print(eccentricity(45))
