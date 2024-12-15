import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'cal_from_0.csv'  # Update with the correct file path
start_line = 50  # Adjust as needed based on observed raw content
data_columns = ["Index", "Date", "Time", "Measurement"]

def load_and_process_data(file_path, start_line):
    laser_data = pd.read_csv(file_path, skiprows=start_line, header=None)
    laser_data.columns = data_columns
    laser_data['Timestamp'] = pd.to_datetime(laser_data['Date'] + ' ' + laser_data['Time'])
    return laser_data

# Load data
laser_data = load_and_process_data(file_path, start_line)

# Plot the full dataset
plt.figure(figsize=(12, 6))
plt.plot(laser_data['Timestamp'], laser_data['Measurement']*1000, label='Laser Measurement (Full Length)', color='blue')
plt.xlabel('Timestamp')
plt.ylabel('intensity [mV]')
plt.title('Laser Measurement from ignition')
# plt.legend()
plt.grid()
plt.show()

# Print measurement duration
start_time = pd.Timestamp('2024-12-02 14:15:18.346')
end_time = pd.Timestamp('2024-12-02 15:15:39.301')
total_duration = end_time - start_time
print(f"Start Time: {start_time}")
print(f"End Time: {end_time}")
print(f"Total Duration: {total_duration}")


print(laser_data['Measurement'].describe())