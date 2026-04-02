'''
Docstring for generate_features_train_test_split
- generate_features_train_test_split.py
    - load every selected window for every experiment 
    - perform filtering on accelerometer and gyroscope signals 
    - save both time series windows and extracted features 
    
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path 
import os 
import re 

window_size = 128 
selected_windows_dir = Path("./selected_windows")

# eg. cycling_1, running_2, jumprope_1, ...
pattern = r'.+_\d+'
folders = [f.name for f in selected_windows_dir.iterdir() if f.is_dir()]
selected_activities = sorted([f for f in folders if re.search(pattern, f)])
selected_activities = [Path(selected_windows_dir, folder) for folder in selected_activities]

# load activity labels 
activity_labels = pd.read_csv(Path("./final_dataset/activity_labels.csv"))
print(activity_labels.head())

# seven parallel dataframes:
# six for the 128 signal window
# one for the labels 
df_X_acc_x = pd.DataFrame(columns = range(window_size)) 
df_X_acc_y = pd.DataFrame(columns = range(window_size)) 
df_X_acc_z = pd.DataFrame(columns = range(window_size)) 
df_X_rot_x = pd.DataFrame(columns = range(window_size)) 
df_X_rot_y = pd.DataFrame(columns = range(window_size)) 
df_X_rot_z = pd.DataFrame(columns = range(window_size)) 
df_y = pd.DataFrame(columns = ["label"])
    
# for each activity 
for selected_activity_dir in selected_activities:
    # Split by the character _ 
    activity_name = str(selected_activity_dir.name)
    activity_type = re.split(r'[_]', activity_name)[0].upper() 
    print(activity_type)
    # get row in df that matches activity type 
    activity_label = activity_labels[activity_labels["activity"] == activity_type]
    # get numeric label corresponding to the activity type 
    activity_label = activity_label["num"].values[0]
    print(activity_label)

    # load all six axis RAW signals
    # the iloc[:,1:] drops the initial time index column
    acc_x_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name}_acc_x.csv")).iloc[:,1:]
    acc_y_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name}_acc_y.csv")).iloc[:,1:]
    acc_z_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name}_acc_z.csv")).iloc[:,1:]
    rot_x_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name}_rot_x.csv")).iloc[:,1:]
    rot_y_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name}_rot_y.csv")).iloc[:,1:]
    rot_z_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name}_rot_z.csv")).iloc[:,1:]

    # get number of windows recorded per activity 
    num_windows = acc_x_df.shape[0]
    
    # make the columns equal so that they can be concatenated.
    df_X_acc_x.columns = acc_x_df.columns 
    df_X_acc_y.columns = acc_y_df.columns 
    df_X_acc_z.columns = acc_z_df.columns 
    df_X_rot_x.columns = rot_x_df.columns 
    df_X_rot_y.columns = rot_y_df.columns 
    df_X_rot_z.columns = rot_z_df.columns 

    # concatenate X dataframes
    # # pd.DataFrame(acc_x_df.to_numpy())
    df_X_acc_x = pd.concat((df_X_acc_x, acc_x_df))
    df_X_acc_y = pd.concat((df_X_acc_y, acc_y_df))
    df_X_acc_z = pd.concat((df_X_acc_z, acc_z_df))
    df_X_rot_x = pd.concat((df_X_rot_x, rot_x_df))
    df_X_rot_y = pd.concat((df_X_rot_y, rot_y_df))
    df_X_rot_z = pd.concat((df_X_rot_z, rot_z_df))
    
    # concatenate y labels 
    new_y_labels = pd.DataFrame([activity_label] * num_windows, columns=df_y.columns)
    df_y = pd.concat([df_y, new_y_labels])

# testing 
print(f"{df_X_acc_x.shape}")
print(f"{df_X_acc_y.shape}")
print(f"{df_X_acc_z.shape}")
print(f"{df_X_rot_x.shape}")
print(f"{df_X_rot_y.shape}")
print(f"{df_X_rot_z.shape}")
print(f"{df_y.shape}")


# ****************************************************************

from scipy.signal import medfilt, butter, filtfilt

'''
final dataset file tree:

final_dataset/ 
    train/
        time_series/
            cycling_1/
                cycling_1_acc_x.csv 
                cycling_1_acc_y.csv
                cycling_1_acc_z.csv  
                cycling_1_rot_x.csv 
                cycling_1_rot_y.csv
                cycling_1_rot_z.csv  
            running_1/
                running_1_acc_x.csv 
                running_1_acc_y.csv
                running_1_acc_z.csv
                running_1_rot_x.csv 
                running_1_rot_y.csv
                running_1_rot_z.csv  
            {experiment_type}_{experiment_index}/
                {experiment_type}_{experiment_index}_acc_x.csv 
                {experiment_type}_{experiment_index}_acc_y.csv
                {experiment_type}_{experiment_index}_acc_z.csv
                {experiment_type}_{experiment_index}_rot_x.csv 
                {experiment_type}_{experiment_index}_rot_y.csv
                {experiment_type}_{experiment_index}_rot_z.csv  
            .
            .
            .
        features/
            cycling_1/
                cycling_1_features.csv 
            running_1/
                running_1_features.csv 
            {experiment_type}_{experiment_index}/
                experiment_type}_{experiment_index}_features.csv
            .
            .
            .
    test/
        **identical to train/**
    activity_labels.csv

algorithm:
- combine_resample.ipynb
    - run once for all experiments 
    - join accelerometer and gyroscope signals and save as one CSV file per experiment
- visualize_combined_df.py 
    - run once per experiment
    - manually select indices per experiment with matplotlib GUI 
    - save CSV files of time series windows per experiment 
        - 6 CSV files per experiment in folders
- generate_features_train_test_split.py
    - perform filtering on accelerometer and gyroscope signals 
    - both time series windows and extracted features 
'''

'''
filepath strings
final_dataset/train/time_series/cycling_1/cycling_1_acc_x.csv 
final_dataset/{train_test}/{time_series_or_features}/{experiment_type}_{experiment_index}/{experiment_type}_{experiment_index}_acc_x.csv 
Path("./final_dataset", )
'''

'''
functions to compute for features 
Butterworth filter 

acc_x, acc_y, acc_z, rot_x, rot_y, rot_z 

The "Big 5" Human-Readable Features

While the full dataset has 561, these five categories usually provide 80-90% of the predictive power for activities like cycling and jumping rope:

    Signal Magnitude Area (SMA): The "Workhorse." It measures the total intensity of movement. High for jumping rope, medium for running, 
    low for cycling (since the phone is relatively stable in a pocket).

    Mean & Standard Deviation: The average "tilt" (Gravity) and the "intensity" of the vibration.

    Energy: The "Power" of the signal. Useful for distinguishing between high-impact running and low-impact cycling.

    Signal Entropy: Measures how "chaotic" the movement is. Walking downstairs is often more "random" than the rhythmic stroke of cycling.

    Dominant Frequency (from FFT): This is your Cadence. It literally tells the model your RPM or strides per minute.
    
time domain
mean
stdev
MAD 
SMA
energy 
IQR
entropy 
correlation 

jerk signals for all channels 

frequency domain 
dominant frequency 
spectral energy 
skewness 
curtosis 
mean frequency 

gravity angles 
'''  

def create_lowpass_filter(cutoff, fs=50, order=3):
    # 2. Apply Low-Pass Butterworth Filter
    # 20 Hz default cutoff frequency
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = butter(order, low, btype='low')
    return b,a
'''
denoise and filter accelerometer signals 
median filter for initial filtering
20 Hz lowpass butterworth filter for high frequency noise
0.3 Hz lowpass butterworth filter for gravity
'''
# accelerometer signal denoises and filters out 
# 
def denoise_accl_signal(data: pd.DataFrame, fs=50, order=3) -> pd.DataFrame: 
    # convert to np.float64
    data = data.to_numpy().astype(np.float64)
    
    nyq = 0.5 * fs

    # 1. Apply Median Filter (kernel size 3 or 5 is standard)
    median_filtered = medfilt(data, kernel_size=3, )

    # 2. Apply Low-Pass Butterworth Filter
    # 20 Hz default cutoff frequency
    cutoff = 20
    b_noise, a_noise = create_lowpass_filter(cutoff, fs, 3)
    # axis = 1 - filter along rows
    total_acc = filtfilt(b_noise, a_noise, median_filtered, axis=1)
    
    # 3. Apply Low-Pass Butterworth Filter
    # 0.3 Hz default cutoff frequency
    cutoff=0.3
    b_grav, a_grav = create_lowpass_filter(cutoff, fs, 3)
    # axis = 1 - filter along rows
    gravity = filtfilt(b_grav, a_grav, total_acc, axis=1)

    # subtract body acceleration from gravity to get total 
    # body_acc = total_acc - gravity

    # return body acceleration and gravity acceleration
    # convert back to DataFrame
    df_total_acc = pd.DataFrame(total_acc)
    df_gravity = pd.DataFrame(gravity)
    
    return df_total_acc, df_gravity 

'''
denoise and filter gyroscope signals 
median filter for initial filtering
20 Hz lowpass butterworth filter for high frequency noise
'''
def denoise_gyro_signal(data, fs=50, order=3):
    # convert to np.float64
    data = data.to_numpy().astype(np.float64)
    
    nyq = 0.5 * fs

    # 1. Apply Median Filter (kernel size 3 or 5 is standard)
    median_filtered = medfilt(data, kernel_size=3)

    # 2. Apply Low-Pass Butterworth Filter
    # 20 Hz default cutoff frequency
    cutoff = 20
    b_noise, a_noise = create_lowpass_filter(cutoff, fs, 3)
    # axis = 1 - filter along rows
    total_rot = filtfilt(b_noise, a_noise, median_filtered, axis=1)

    # convert back to DataFrame 
    df_total_rot = pd.DataFrame(total_rot)
    
    return df_total_rot 

# ****************************************************************
# final dataset dir 
final_dataset_path = Path("./final_dataset/")
train_dir = Path("train/")
test_dir = Path("test/")
time_series = Path("time_series/")
features = Path("features")
# final_dataset/{train_test}/{time_series_or_features}/{experiment_type}_{experiment_index}/{experiment_type}_{experiment_index}_acc_x.csv 

# perform a train-test-split on the indices 
indices = np.arange(len(df_y))

# 2. Split the indices, NOT the data yet
# Stratify=y_labels ensures each activity is represented 80/20
idx_train, idx_test = train_test_split(
    indices, 
    test_size=0.2, 
    stratify=df_y, 
    random_state=42
)

# at this point the indices for train-test-split have been created. 
print(idx_train, idx_test)

# filter the data 
df_acc_x_total, df_acc_x_gravity = denoise_accl_signal(df_X_acc_x)
df_acc_y_total, df_acc_y_gravity = denoise_accl_signal(df_X_acc_y)
df_acc_z_total, df_acc_z_gravity = denoise_accl_signal(df_X_acc_z)
df_rot_x_total = denoise_gyro_signal(df_X_rot_x)
df_rot_y_total = denoise_gyro_signal(df_X_rot_y)
df_rot_z_total = denoise_gyro_signal(df_X_rot_z)

# at this point the data has been filtered. 

# ****************************************************************
# Train-Test-Split 



# ****************************************************************
# Feature Extraction
# calculate features

'''
mean body acc xyz 
mean gravity acc xyz 
mean rot xyz

stdev body acc xyz 
'''
import numpy as np
from scipy.fftpack import fft
from scipy.stats import entropy

df_X_features = pd.DataFrame()

# mean 
df_X_features["acc_x_total_mean"] = df_acc_x_total.mean(axis=1)
df_X_features["acc_y_total_mean"] = df_acc_y_total.mean(axis=1)
df_X_features["acc_z_total_mean"] = df_acc_z_total.mean(axis=1)
df_X_features["acc_x_gravity_mean"] = df_acc_x_gravity.mean(axis=1)
df_X_features["acc_y_gravity_mean"] = df_acc_y_gravity.mean(axis=1)
df_X_features["acc_z_gravity_mean"] = df_acc_z_gravity.mean(axis=1)
df_X_features["rot_x_total_mean"] = df_rot_x_total.mean(axis=1)
df_X_features["rot_y_total_mean"] = df_rot_y_total.mean(axis=1)
df_X_features["rot_z_total_mean"] = df_rot_z_total.mean(axis=1)
# standard deviation
df_X_features["acc_x_total_std"] = df_acc_x_total.std(axis=1)
df_X_features["acc_y_total_std"] = df_acc_y_total.std(axis=1)
df_X_features["acc_z_total_std"] = df_acc_z_total.std(axis=1)
df_X_features["acc_x_gravity_std"] = df_acc_x_gravity.std(axis=1)
df_X_features["acc_y_gravity_std"] = df_acc_y_gravity.std(axis=1)
df_X_features["acc_z_gravity_std"] = df_acc_z_gravity.std(axis=1)
df_X_features["rot_x_total_std"] = df_rot_x_total.std(axis=1)
df_X_features["rot_y_total_std"] = df_rot_y_total.std(axis=1)
df_X_features["rot_z_total_std"] = df_rot_z_total.std(axis=1)
# min
df_X_features["acc_x_total_min"] = df_acc_x_total.min(axis=1)
df_X_features["acc_y_total_min"] = df_acc_y_total.min(axis=1)
df_X_features["acc_z_total_min"] = df_acc_z_total.min(axis=1)
df_X_features["acc_x_gravity_min"] = df_acc_x_gravity.min(axis=1)
df_X_features["acc_y_gravity_min"] = df_acc_y_gravity.min(axis=1)
df_X_features["acc_z_gravity_min"] = df_acc_z_gravity.min(axis=1)
df_X_features["rot_x_total_min"] = df_rot_x_total.min(axis=1)
df_X_features["rot_y_total_min"] = df_rot_y_total.min(axis=1)
df_X_features["rot_z_total_min"] = df_rot_z_total.min(axis=1)
# max
df_X_features["acc_x_total_max"] = df_acc_x_total.max(axis=1)
df_X_features["acc_y_total_max"] = df_acc_y_total.max(axis=1)
df_X_features["acc_z_total_max"] = df_acc_z_total.max(axis=1)
df_X_features["acc_x_gravity_max"] = df_acc_x_gravity.max(axis=1)
df_X_features["acc_y_gravity_max"] = df_acc_y_gravity.max(axis=1)
df_X_features["acc_z_gravity_max"] = df_acc_z_gravity.max(axis=1)
df_X_features["rot_x_total_max"] = df_rot_x_total.max(axis=1)
df_X_features["rot_y_total_max"] = df_rot_y_total.max(axis=1)
df_X_features["rot_z_total_max"] = df_rot_z_total.max(axis=1)
# # magnitude - not a feature
# df_X_features["acc_total_mag"] = np.sqrt(np.square(df_acc_x_total) + np.square(df_acc_y_total) + np.square(df_acc_z_total))
# df_X_features["acc_gravity_mag"] = np.sqrt(np.square(df_acc_x_gravity) + np.square(df_acc_y_gravity) + np.square(df_acc_z_gravity))
# df_X_features["rot_total_mag"] = np.sqrt(np.square(df_rot_x_total) + np.square(df_rot_y_total) + np.square(df_rot_z_total))
# # median
df_X_features["acc_x_total_median"] = df_acc_x_total.median(axis=1)
df_X_features["acc_y_total_median"] = df_acc_y_total.median(axis=1)
df_X_features["acc_z_total_median"] = df_acc_z_total.median(axis=1)
df_X_features["acc_x_gravity_median"] = df_acc_x_gravity.median(axis=1)
df_X_features["acc_y_gravity_median"] = df_acc_y_gravity.median(axis=1)
df_X_features["acc_z_gravity_median"] = df_acc_z_gravity.median(axis=1)
df_X_features["rot_x_total_median"] = df_rot_x_total.median(axis=1)
df_X_features["rot_y_total_median"] = df_rot_y_total.median(axis=1)
df_X_features["rot_z_total_median"] = df_rot_z_total.median(axis=1)
# # mad
def get_mad(data):
    return np.median(data.to_numpy() - np.median(data.to_numpy(), axis=1).reshape(-1, 1), axis=1)
df_X_features["acc_x_total_mad"] = get_mad(df_acc_x_total)
df_X_features["acc_y_total_mad"] = get_mad(df_acc_y_total)
df_X_features["acc_z_total_mad"] = get_mad(df_acc_z_total)
df_X_features["acc_x_gravity_mad"] = get_mad(df_acc_x_gravity)
df_X_features["acc_y_gravity_mad"] = get_mad(df_acc_y_gravity)
df_X_features["acc_z_gravity_mad"] = get_mad(df_acc_z_gravity)
df_X_features["rot_x_total_mad"] = get_mad(df_rot_x_total)
df_X_features["rot_y_total_mad"] = get_mad(df_rot_y_total)
df_X_features["rot_z_total_mad"] = get_mad(df_rot_z_total)
# sma
def get_sma(data_x, data_y, data_z):
    return np.mean(np.abs(data_x.to_numpy()) + np.abs(data_y.to_numpy()) + np.abs(data_z.to_numpy()), axis=1)
df_X_features["acc_total_sma"] = get_sma(df_acc_x_total, df_acc_y_total, df_acc_z_total)
df_X_features["acc_gravity_sma"] = get_sma(df_acc_x_gravity, df_acc_y_gravity, df_acc_z_gravity)
df_X_features["rot_total_sma"] = get_sma(df_rot_x_total, df_rot_y_total, df_rot_z_total)
# energy
def get_energy(data):
    return np.mean(np.square(data.to_numpy()), axis=1)
df_X_features["acc_x_total_energy"] = get_energy(df_acc_x_total)
df_X_features["acc_y_total_energy"] = get_energy(df_acc_y_total)
df_X_features["acc_z_total_energy"] = get_energy(df_acc_z_total)
df_X_features["acc_x_gravity_energy"] = get_energy(df_acc_x_gravity)
df_X_features["acc_y_gravity_energy"] = get_energy(df_acc_y_gravity)
df_X_features["acc_z_gravity_energy"] = get_energy(df_acc_z_gravity)
df_X_features["rot_x_total_energy"] = get_energy(df_rot_x_total)
df_X_features["rot_y_total_energy"] = get_energy(df_rot_y_total)
df_X_features["rot_z_total_energy"] = get_energy(df_rot_z_total)
# # iqr

# df_X_features["acc_x_total_"] = df_acc_x_total.max()
# df_X_features["acc_y_total_"] = df_acc_y_total.
# df_X_features["acc_z_total_"] = df_acc_z_total.
# df_X_features["acc_x_gravity_"] = df_acc_x_gravity.
# df_X_features["acc_y_gravity_"] = df_acc_y_gravity.
# df_X_features["acc_z_gravity_"] = df_acc_z_gravity.
# df_X_features["rot_x_total_"] = df_rot_x_total.
# df_X_features["rot_y_total_"] = df_rot_y_total.
# df_X_features["rot_z_total_"] = df_rot_z_total.
# entropy 
def compute_entropy(data):
    data = data.to_numpy()
    # 1. Perform FFT and get the magnitude (absolute value)
    fft_coeffs = np.fft.fft(data, axis=1)
    psd = np.abs(fft_coeffs)**2
    
    # 2. Normalize the PSD so it sums to 1 (like a probability distribution)
    # Avoid division by zero with a small epsilon
    psd_sum = np.sum(psd, axis=1, keepdims=True)
    psd_norm = psd / (psd_sum + 1e-9)
    
    # 3. Calculate Shannon Entropy
    # We use a mask to avoid log(0) which is undefined
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-9), axis=1)
    
    return entropy
df_X_features["acc_x_total_entropy"] = compute_entropy(df_acc_x_total)
df_X_features["acc_y_total_entropy"] = compute_entropy(df_acc_y_total)
df_X_features["acc_z_total_entropy"] = compute_entropy(df_acc_z_total)
df_X_features["acc_x_gravity_entropy"] = compute_entropy(df_acc_x_gravity)
df_X_features["acc_y_gravity_entropy"] = compute_entropy(df_acc_y_gravity)
df_X_features["acc_z_gravity_entropy"] = compute_entropy(df_acc_z_gravity)
df_X_features["rot_x_total_entropy"] = compute_entropy(df_rot_x_total)
df_X_features["rot_y_total_entropy"] = compute_entropy(df_rot_y_total)
df_X_features["rot_z_total_entropy"] = compute_entropy(df_rot_z_total)
# dominant frequencies 
def get_dominant_frequencies(windowed_data, sampling_rate=50):
    """
    Finds the frequency with the highest power for each window.
    sampling_rate: The Hz of your ESP32 data collection (e.g., 50Hz or 100Hz).
    """
    n = windowed_data.shape[1]  # 128 samples
    
    # 1. Compute FFT
    fft_values = np.fft.fft(windowed_data, axis=1)
    
    # 2. Get Power (Magnitudes)
    # We only care about the first half (0 to fs/2) because FFT is symmetric
    num_useful_bins = n // 2
    psd = np.abs(fft_values[:, :num_useful_bins])
    
    # 3. Find index of the maximum power for each window
    # for each window
    # We ignore the 0th bin (DC component/Offset) as it's usually just gravity/bias
    peak_indices = np.argmax(psd[:, 1:], axis=1) + 1 
    
    # 4. Convert index to Hz
    # Formula: Frequency = (index * sampling_rate) / total_samples
    freqs = (peak_indices * sampling_rate) / n
    
    return freqs
df_X_features["acc_x_total_dominant_frequency"] = get_dominant_frequencies(df_acc_x_total)
df_X_features["acc_y_total_dominant_frequency"] = get_dominant_frequencies(df_acc_y_total)
df_X_features["acc_z_total_dominant_frequency"] = get_dominant_frequencies(df_acc_z_total)
df_X_features["acc_x_gravity_dominant_frequency"] = get_dominant_frequencies(df_acc_x_gravity)
df_X_features["acc_y_gravity_dominant_frequency"] = get_dominant_frequencies(df_acc_y_gravity)
df_X_features["acc_z_gravity_dominant_frequency"] = get_dominant_frequencies(df_acc_z_gravity)
df_X_features["rot_x_total_dominant_frequency"] = get_dominant_frequencies(df_rot_x_total)
df_X_features["rot_y_total_dominant_frequency"] = get_dominant_frequencies(df_rot_y_total)
df_X_features["rot_z_total_dominant_frequency"] = get_dominant_frequencies(df_rot_z_total)
# mean gravity
df_X_features["acc_gravity_mean"] = (df_acc_x_gravity + df_acc_y_gravity + df_acc_z_gravity).mean(axis=1)/3
# 
# df_X_features["acc_x_total_"] = df_acc_x_total.max()
# df_X_features["acc_y_total_"] = df_acc_y_total.
# df_X_features["acc_z_total_"] = df_acc_z_total.
# df_X_features["acc_x_gravity_"] = df_acc_x_gravity.
# df_X_features["acc_y_gravity_"] = df_acc_y_gravity.
# df_X_features["acc_z_gravity_"] = df_acc_z_gravity.
# df_X_features["rot_x_total_"] = df_rot_x_total.
# df_X_features["rot_y_total_"] = df_rot_y_total.
# df_X_features["rot_z_total_"] = df_rot_z_total.

# df_X_features["acc_x_total_"] = df_acc_x_total.max()
# df_X_features["acc_y_total_"] = df_acc_y_total.
# df_X_features["acc_z_total_"] = df_acc_z_total.
# df_X_features["acc_x_gravity_"] = df_acc_x_gravity.
# df_X_features["acc_y_gravity_"] = df_acc_y_gravity.
# df_X_features["acc_z_gravity_"] = df_acc_z_gravity.
# df_X_features["rot_x_total_"] = df_rot_x_total.
# df_X_features["rot_y_total_"] = df_rot_y_total.
# df_X_features["rot_z_total_"] = df_rot_z_total.

pass

# df_X_features[""] = 











# def extract_uci_features(window_data):
#     """
#     window_data: A (128, 6) array [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
#     """
#     features = []
    
#     # Separate Acc and Gyro for processing
#     acc = window_data[:, :3]
#     gyro = window_data[:, 3:]
    
#     # 1. MEAN (6 features)
#     features.extend(np.mean(window_data, axis=0))
    
#     # 2. STD DEV (6 features)
#     features.extend(np.std(window_data, axis=0))
    
#     # 3. SMA (Signal Magnitude Area) - 1 feature
#     # Sum of absolute values of 3-axis acceleration
#     sma = np.sum(np.abs(acc)) / 128
#     features.append(sma)
    
#     # 4. ENERGY (6 features)
#     # Sum of the squares divided by the window size
#     energy = np.sum(window_data**2, axis=0) / 128
#     features.extend(energy)
    
#     # 5. ENTROPY (6 features)
#     # Measure of complexity using a histogram-based approach
#     for i in range(6):
#         hist, _ = np.histogram(window_data[:, i], bins=10, density=True)
#         features.append(entropy(hist + 1e-9)) # Add small epsilon to avoid log(0)
        
#     # 6. DOMINANT FREQUENCY (6 features)
#     for i in range(6):
#         # Perform Fast Fourier Transform
#         sig_fft = np.abs(fft(window_data[:, i]))
#         # We only care about the first half (positive frequencies)
#         dominant_freq_index = np.argmax(sig_fft[1:64]) # Skip DC component at index 0
#         features.append(dominant_freq_index)
        
#     return np.array(features)

# # Process the entire ride into a Feature Matrix
# window_size = 128
# hop_size = 64
# feature_matrix = []

# for i in range(0, len(df_resampled) - window_size, hop_size):
#     window = df_resampled.iloc[i : i + window_size].values
#     vector = extract_uci_features(window)
#     feature_matrix.append(vector)

# X_features = np.array(feature_matrix)
# print(f"Final Feature Matrix Shape: {X_features.shape}") # Should be (N_windows, 31)






from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_har_model(input_shape, num_classes):
    model = Sequential([
        # First Convolutional Layer
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        
        # Second Convolutional Layer
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax') # Softmax for multi-class
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Initialize (6 axes, 5 activities)
model = build_har_model((128, 9), 5)
model.summary()






# import numpy as np
# import pandas as pd

def load_dataset_to_3d(file_prefix='signal_', axes=None):
    if axes is None:
        axes = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    instance_list = []
    
    for axis in axes:
        # Load the 128-column CSV (header=None since we saved without headers)
        df = pd.read_csv(f'{file_prefix}{axis}.csv', header=None)
        # Convert to numpy and add a "depth" dimension
        # Shape change: (N, 128) -> (N, 128, 1)
        instance_list.append(df.values[:, :, np.newaxis])
    
    # Stack along the last axis to get (N, 128, 6)
    X = np.concatenate(instance_list, axis=-1)
    
    # Load labels
    y = pd.read_csv('y_labels.csv', header=None).values
    
    return X, y

# # Execute
# X_train, y_train = load_dataset_to_3d()

# print(f"X_train shape: {X_train.shape}") # Expected: (Samples, 128, 6)
# print(f"y_train shape: {y_train.shape}") # Expected: (Samples, 1)
















# import pandas as pd
# import numpy as np

# def generate_uci_signals(df_source, indices, window_size=128):
#     axes = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
#     signal_dfs = {}
    
#     for axis in axes:
#         rows = []
#         for start_idx in indices:
#             window = df_source[axis].iloc[start_idx : start_idx + window_size].values
#             if len(window) == window_size:
#                 rows.append(window)
        
#         # Create 128-column DataFrame
#         col_names = [f't{i}' for i in range(window_size)]
#         signal_dfs[axis] = pd.DataFrame(rows, columns=col_names)
        
#     return signal_dfs

# # Execute
# all_signals = generate_uci_signals(df_resampled, marked_indices)

# # Save each axis to its own CSV
# for axis, df_wide in all_signals.items():
#     df_wide.to_csv(f'signal_{axis}.csv', index=False, header=False)
#     print(f"Saved {axis} with shape {df_wide.shape}")

# # Create and save the activity label CSV
# # Assuming all indices in this session belong to one activity (e.g., 'Cycling' = 4)
# activity_id = 4 
# y_labels = pd.DataFrame([activity_id] * len(marked_indices), columns=['activity'])
# y_labels.to_csv('y_labels.csv', index=False, header=False)















# 1. Signal Magnitude Area (SMA)

# The "Intensity" Meter. SMA is the sum of the absolute values of the three axes.

#     Jumping Rope: Massive SMA. Every jump creates a huge spike across all axes.

#     Cycling: Very low SMA. Since the phone is in your pocket, the "impact" is minimal; it's mostly smooth rotation.

#     Running: High SMA, but usually lower and more consistent than jumping rope.

# 2. Dominant Frequency (via FFT)

# The "Cadence" Meter.
# By performing a Fast Fourier Transform (FFT) on your body_acc windows, you find the frequency with the most power.

#     Cycling: A clear peak around 1.2–1.6 Hz (70–95 RPM).

#     Running: A peak around 2.5–3.0 Hz (150–180 steps per minute).

#     Jumping Rope: Very high frequency peaks (3 Hz+ depending on your speed).

# 3. Mean Gravity (The "Tilt" Feature)

# The "Slope" Meter.
# By taking the mean of your grav_acc signals, you get the average orientation of the phone.

#     Going Upstairs/Downstairs: Your body naturally leans forward (pitch change). The gravity vector shifts significantly compared to level-ground running.

#     Cycling: Because the phone is in your pocket, the gravity vector "oscillates" as your thigh moves up and down. The range of the gravity signal is a dead giveaway for cycling.

# 4. Signal Entropy

# The "Complexity" Meter.
# Entropy measures how "ordered" a signal is.

#     Running/Cycling: Low Entropy. These are highly rhythmic and predictable.

#     Going Downstairs: High Entropy. Gravity and impact patterns are often more "chaotic" as you stabilize your weight on each step.

# 5. Jerk Signals (The "Suddenness" Feature)

# The "Snap" Meter.
# Jerk is the derivative of acceleration (da/dt). It measures how quickly you change your acceleration.

#     Jumping Rope: High Jerk. The transition from "falling" to "rebounding" happens in milliseconds.

#     Cycling: Very Low Jerk. The movement is fluid and continuous.








import numpy as np
import pandas as pd

def calculate_derivatives_and_mag(df):
    """
    Adds Jerk and Magnitude columns to your processed dataframe.
    Assumes 50Hz (dt = 0.02s)
    """
    dt = 0.02
    new_df = df.copy()
    
    # 1. Calculate Magnitudes
    for prefix in ['body_acc', 'grav_acc', 'clean_gyro']:
        cols = [f'{prefix}_x', f'{prefix}_y', f'{prefix}_z']
        new_df[f'{prefix}_mag'] = np.sqrt((df[cols]**2).sum(axis=1))
    
    # 2. Calculate Jerk (Derivative of Body Acc and Gyro)
    # We use np.gradient for a more stable central difference
    for axis in ['x', 'y', 'z', 'mag']:
        # Body Acc Jerk
        new_df[f'body_acc_jerk_{axis}'] = np.gradient(new_df[f'body_acc_{axis}'], dt)
        # Gyro Jerk (Angular Acceleration)
        new_df[f'gyro_jerk_{axis}'] = np.gradient(new_df[f'clean_gyro_{axis}'], dt)
        
    return new_df

# # Usage:
# # df_augmented = calculate_derivatives_and_mag(df_final)











# unique, counts = np.unique(y_test, return_counts=True)
# print("Test set distribution:", dict(zip(unique, counts)))












# from sklearn.preprocessing import StandardScaler

# # 1. Flatten your 3D training data to 2D to fit the scaler
# # Shape: (N_train, 128, 12) -> (N_train * 128, 12)
# N, T, C = X_train.shape
# X_train_reshaped = X_train.reshape(-1, C)

# scaler = StandardScaler()
# X_train_normalized = scaler.fit_transform(X_train_reshaped)

# # 2. Reshape back to 3D
# X_train = X_train_normalized.reshape(N, T, C)

# # 3. Apply the SAME scaler to the Test set (DO NOT call fit_transform)
# N_t, T_t, C_t = X_test.shape
# X_test_reshaped = X_test.reshape(-1, C_t)
# X_test_normalized = scaler.transform(X_test_reshaped)
# X_test = X_test_normalized.reshape(N_t, T_t, C_t)






# 1. The Golden Rule: Order of Operations

# The sequence should always be:

#     Denoise & Filter (Continuous Data)

#     Segment into Windows (Overlap 50%)

#     Stratified Train-Test Split (Indices)

#     Calculate Features (Optional: For Random Forest)

#     Normalize (Fit on Train, Transform Test)









import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        # 1. First Convolutional Block
        # Filters=32, Kernel=3 is standard for 50Hz signals
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        
        # 2. Second Convolutional Block
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # 3. Flatten and Regularization
        Flatten(),
        Dropout(0.5), # Crucial because your dataset is small
        
        # 4. Fully Connected Output
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax') # Softmax for probability distribution
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', # Handles your 0-4 integer labels
        metrics=['accuracy']
    )
    
    return model

# # Your input shape: (128 time steps, 12 channels)
# model = build_cnn_model((128, 12), 5)
# model.summary()
















# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# # 1. Initialize K-Fold (5 folds means 80/20 split each time)
# skf = StratifiedKFold(n_sequences=5, shuffle=True, random_state=42)

# fold_accuracies = []

# # 2. Start the Cross-Validation Loop
# # We use 'y' to ensure each activity is represented equally in every fold
# for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
#     print(f"--- Training Fold {fold+1} ---")
    
#     # A. Split the data using the indices
#     X_train, X_test = X_raw[train_idx], X_raw[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
    
#     # B. Normalize (CRITICAL: Fit on Train, Transform Test to avoid leakage)
#     N, T, C = X_train.shape
#     scaler = StandardScaler()
#     X_train_reshaped = scaler.fit_transform(X_train.reshape(-1, C)).reshape(N, T, C)
    
#     N_t, T_t, C_t = X_test.shape
#     X_test_reshaped = scaler.transform(X_test.reshape(-1, C_t)).reshape(N_t, T_t, C_t)
    
#     # C. Build and Train your 1D-CNN
#     model = build_cnn_model((128, 12), 5)
#     model.fit(X_train_reshaped, y_train, epochs=30, batch_size=16, verbose=0)
    
#     # D. Evaluate
#     loss, acc = model.evaluate(X_test_reshaped, y_test, verbose=0)
#     fold_accuracies.append(acc)
#     print(f"Fold {fold+1} Accuracy: {acc:.4f}")

# # 3. Final Result
# print(f"\nFinal Cross-Validation Accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")




# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score

# # Load feature names
# features = pd.read_csv('UCI HAR Dataset/features.txt', sep='\s+', header=None, names=['ID', 'Name'])

# # Load Training Data
# X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', sep='\s+', header=None)
# y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None)

# # Load Test Data
# X_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', sep='\s+', header=None)
# y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None)

# # Assign feature names to columns (optional but helpful for feature importance)
# X_train.columns = features['Name']
# X_test.columns = features['Name']

# # Initialize the classifier
# # n_estimators=100 is a good start; random_state ensures reproducibility
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# # Train the model
# rf_model.fit(X_train, y_train.values.ravel())

# # Make predictions
# y_pred = rf_model.predict(X_test)

# # Check results
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print(classification_report(y_test, y_pred))




