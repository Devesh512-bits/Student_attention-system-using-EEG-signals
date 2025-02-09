import serial
import serial.tools.list_ports
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import MinMaxScaler
import re

# Function to find available serial ports
def find_serial_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"Found port: {port.device}")
    return ports[0].device if ports else None

# Configure Serial Connection
SERIAL_PORT = find_serial_port()  # Automatically detect the first available port
BAUD_RATE = 115200

# Check Serial Port
if SERIAL_PORT:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
else:
    print("No available serial ports detected. Please check your connection.")
    exit()

# Read data from serial port
def read_serial_data():
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        print(f"Received: {line}")  # Debugging step
        if re.match(r'^-?\d+(\.\d+)?$', line):  # Validate numeric data
            return float(line)
        return None
    except Exception as e:
        print(f"Error reading serial: {e}")
        return None

# Bandpass filter for EEG signal
def bandpass_filter(data, lowcut=0.5, highcut=30, fs=256, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Notch filter to remove power line interference
def notch_filter(data, freq=50.0, fs=256, Q=30):
    nyquist = 0.5 * fs
    notch_freq = freq / nyquist
    b, a = iirnotch(notch_freq, Q)
    return filtfilt(b, a, data)

# Moving average smoothing
def smooth_data(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Initialize variables
data_buffer = []
time_buffer = []

# Setup Matplotlib for Live Plot
fig, ax = plt.subplots()
ax.set_title("Live EEG Signal")
ax.set_xlabel("Time")
ax.set_ylabel("EEG Amplitude")
(line,) = ax.plot([], [], label='EEG Signal', color='blue')
ax.legend()

# Update function for animation
def update(frame):
    data = read_serial_data()
    timestamp = datetime.datetime.now()

    if data is not None:
        data_buffer.append(data)
        time_buffer.append(timestamp.strftime('%H:%M:%S'))

        if len(data_buffer) > 100:  # Limit to 100 points for smooth plotting
            data_buffer.pop(0)
            time_buffer.pop(0)

        # Apply smoothing for better visualization
        smoothed_data = smooth_data(np.array(data_buffer)) if len(data_buffer) > 5 else data_buffer

        line.set_xdata(range(len(smoothed_data)))
        line.set_ydata(smoothed_data)
        ax.relim()
        ax.autoscale_view()

ani = animation.FuncAnimation(fig, update, interval=50)

# Collect data for a fixed duration or number of samples
print("Recording EEG data...")
try:
    plt.show()
except KeyboardInterrupt:
    print("Recording stopped.")
finally:
    ser.close()
    if data_buffer:
        df = pd.DataFrame(zip(time_buffer, data_buffer), columns=["Timestamp", "EEG Signal"])

        # Apply Bandpass Filter
        filtered_data = bandpass_filter(df["EEG Signal"].values)
        # Apply Notch Filter
        filtered_data = notch_filter(filtered_data)
        
        # Normalize EEG Data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(filtered_data.reshape(-1, 1)).flatten()

        # Save to DataFrame
        df["Filtered EEG"] = filtered_data
        df["Normalized EEG"] = normalized_data

        # Save CSV
        filename = f"eeg_preprocessed_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"EEG data saved to {filename}")
    else:
        print("No valid EEG data recorded.")
