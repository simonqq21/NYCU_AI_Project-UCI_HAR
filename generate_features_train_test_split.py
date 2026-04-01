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

time domain
mean
stdev
MAD 
SMA
energy 
IQR
entropy 
correlation 

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
# Feature Extraction
# calculate features

# df_features = pd.DataFrame()