'''
Docstring for generate_features_train_test_split
- generate_features_train_test_split.py
    - load every selected window for every experiment 
    - perform filtering on accelerometer and gyroscope signals 
    - save both time series windows and extracted features 

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

# ****************************************************************
# load activity labels 
activity_labels = pd.read_csv(Path("./final_dataset/activity_labels.csv"))
print(activity_labels.head())

# ****************************************************************
# OUTPUT files 
# Load all windows into seven parallel dataframes
# 22 dataframes
# 9 dfs for X train and test time series
# 1 df each for X train and test features 
# 1 df each for y train and test labels 
df_X_acc_x_total_train = pd.DataFrame(columns = range(window_size)) 
df_X_acc_y_total_train = pd.DataFrame(columns = range(window_size)) 
df_X_acc_z_total_train = pd.DataFrame(columns = range(window_size)) 
df_X_acc_x_gravity_train = pd.DataFrame(columns = range(window_size)) 
df_X_acc_y_gravity_train = pd.DataFrame(columns = range(window_size)) 
df_X_acc_z_gravity_train = pd.DataFrame(columns = range(window_size)) 
df_X_rot_x_total_train = pd.DataFrame(columns = range(window_size)) 
df_X_rot_y_total_train = pd.DataFrame(columns = range(window_size)) 
df_X_rot_z_total_train = pd.DataFrame(columns = range(window_size)) 

df_X_features_train = pd.DataFrame()

df_y_train = pd.DataFrame(columns = ["label"])


df_X_acc_x_total_test = pd.DataFrame(columns = range(window_size)) 
df_X_acc_y_total_test = pd.DataFrame(columns = range(window_size)) 
df_X_acc_z_total_test = pd.DataFrame(columns = range(window_size)) 
df_X_acc_x_gravity_test = pd.DataFrame(columns = range(window_size)) 
df_X_acc_y_gravity_test = pd.DataFrame(columns = range(window_size)) 
df_X_acc_z_gravity_test = pd.DataFrame(columns = range(window_size)) 
df_X_rot_x_total_test = pd.DataFrame(columns = range(window_size)) 
df_X_rot_y_total_test = pd.DataFrame(columns = range(window_size)) 
df_X_rot_z_total_test = pd.DataFrame(columns = range(window_size)) 

df_X_features_test = pd.DataFrame()

df_y_test = pd.DataFrame(columns = ["label"])

# ****************************************************************
folders = [f.name for f in selected_windows_dir.iterdir() if f.is_dir()]
selected_activities = sorted([f for f in folders if re.search(pattern, f)])
selected_activities = [Path(selected_windows_dir, folder) for folder in selected_activities]
# for each activity dir
for selected_activity_dir in selected_activities:
    # Split by the character _ 
    activity_name = str(selected_activity_dir.name)
    
    split_activity_name = re.split(r'[_]', activity_name)
    activity_name_index = f"{split_activity_name[0]}_{split_activity_name[1]}"
    activity_type = split_activity_name[0].upper() 
    activity_train_test = split_activity_name[2]
    print(activity_type, activity_train_test)
    # get row in df that matches activity type 
    activity_label = activity_labels[activity_labels["activity"] == activity_type]
    # get numeric label corresponding to the activity type 
    activity_label = activity_label["num"].values[0]
    print(activity_label)

    # load all nine axis RAW signals
    # the iloc[:,1:] drops the initial time index column
    try:    
        acc_x_total_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name_index}_acc_x_total_{activity_train_test}.csv")).iloc[:,1:]
        acc_y_total_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name_index}_acc_y_total_{activity_train_test}.csv")).iloc[:,1:]
        acc_z_total_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name_index}_acc_z_total_{activity_train_test}.csv")).iloc[:,1:]
        acc_x_gravity_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name_index}_acc_x_gravity_{activity_train_test}.csv")).iloc[:,1:]
        acc_y_gravity_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name_index}_acc_y_gravity_{activity_train_test}.csv")).iloc[:,1:]
        acc_z_gravity_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name_index}_acc_z_gravity_{activity_train_test}.csv")).iloc[:,1:]

        # get body acc from total acceleration and gravity acceleration
        acc_x_total_df = acc_x_total_df - acc_x_gravity_df
        acc_y_total_df = acc_y_total_df - acc_y_gravity_df
        acc_z_total_df = acc_z_total_df - acc_z_gravity_df
        
        rot_x_total_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name_index}_rot_x_total_{activity_train_test}.csv")).iloc[:,1:]
        rot_y_total_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name_index}_rot_y_total_{activity_train_test}.csv")).iloc[:,1:]
        rot_z_total_df = pd.read_csv(Path(selected_activity_dir, f"{activity_name_index}_rot_z_total_{activity_train_test}.csv")).iloc[:,1:]

        # get number of windows recorded per activity 
        num_windows = acc_x_total_df.shape[0]
        # concatenate X dataframes
        if activity_train_test == "test":
            # make the columns equal so that they can be concatenated.
            df_X_acc_x_total_test.columns = acc_x_total_df.columns 
            df_X_acc_y_total_test.columns = acc_y_total_df.columns 
            df_X_acc_z_total_test.columns = acc_z_total_df.columns 
            df_X_acc_x_gravity_test.columns = acc_x_total_df.columns 
            df_X_acc_y_gravity_test.columns = acc_y_total_df.columns 
            df_X_acc_z_gravity_test.columns = acc_z_total_df.columns 
            df_X_rot_x_total_test.columns = rot_x_total_df.columns 
            df_X_rot_y_total_test.columns = rot_y_total_df.columns 
            df_X_rot_z_total_test.columns = rot_z_total_df.columns 
        
            df_X_acc_x_total_test = pd.concat((df_X_acc_x_total_test, acc_x_total_df))
            df_X_acc_y_total_test = pd.concat((df_X_acc_y_total_test, acc_y_total_df))
            df_X_acc_z_total_test = pd.concat((df_X_acc_z_total_test, acc_z_total_df))
            df_X_acc_x_gravity_test = pd.concat((df_X_acc_x_gravity_test, acc_x_total_df))
            df_X_acc_y_gravity_test = pd.concat((df_X_acc_y_gravity_test, acc_y_total_df))
            df_X_acc_z_gravity_test = pd.concat((df_X_acc_z_gravity_test, acc_z_total_df))
            df_X_rot_x_total_test = pd.concat((df_X_rot_x_total_test, rot_x_total_df))
            df_X_rot_y_total_test = pd.concat((df_X_rot_y_total_test, rot_y_total_df))
            df_X_rot_z_total_test = pd.concat((df_X_rot_z_total_test, rot_z_total_df))
            
            # concatenate y labels 
            new_y_labels = pd.DataFrame([activity_label] * num_windows, columns=df_y_test.columns)
            df_y_test = pd.concat([df_y_test, new_y_labels])
        else:
            # make the columns equal so that they can be concatenated.
            df_X_acc_x_total_train.columns = acc_x_total_df.columns 
            df_X_acc_y_total_train.columns = acc_y_total_df.columns 
            df_X_acc_z_total_train.columns = acc_z_total_df.columns 
            df_X_acc_x_gravity_train.columns = acc_x_total_df.columns 
            df_X_acc_y_gravity_train.columns = acc_y_total_df.columns 
            df_X_acc_z_gravity_train.columns = acc_z_total_df.columns 
            df_X_rot_x_total_train.columns = rot_x_total_df.columns 
            df_X_rot_y_total_train.columns = rot_y_total_df.columns 
            df_X_rot_z_total_train.columns = rot_z_total_df.columns 
        
            df_X_acc_x_total_train = pd.concat((df_X_acc_x_total_train, acc_x_total_df))
            df_X_acc_y_total_train = pd.concat((df_X_acc_y_total_train, acc_y_total_df))
            df_X_acc_z_total_train = pd.concat((df_X_acc_z_total_train, acc_z_total_df))
            df_X_acc_x_gravity_train = pd.concat((df_X_acc_x_gravity_train, acc_x_total_df))
            df_X_acc_y_gravity_train = pd.concat((df_X_acc_y_gravity_train, acc_y_total_df))
            df_X_acc_z_gravity_train = pd.concat((df_X_acc_z_gravity_train, acc_z_total_df))
            df_X_rot_x_total_train = pd.concat((df_X_rot_x_total_train, rot_x_total_df))
            df_X_rot_y_total_train = pd.concat((df_X_rot_y_total_train, rot_y_total_df))
            df_X_rot_z_total_train = pd.concat((df_X_rot_z_total_train, rot_z_total_df))
            
            # concatenate y labels 
            new_y_labels = pd.DataFrame([activity_label] * num_windows, columns=df_y_train.columns)
            df_y_train = pd.concat([df_y_train, new_y_labels])

    except FileNotFoundError:
        print("empty folder")

# testing
def print_dataset_size(): 
    print(f"{df_X_acc_x_total_train.shape}")
    print(f"{df_X_acc_y_total_train.shape}")
    print(f"{df_X_acc_z_total_train.shape}")
    print(f"{df_X_acc_x_gravity_train.shape}")
    print(f"{df_X_acc_y_gravity_train.shape}")
    print(f"{df_X_acc_z_gravity_train.shape}")
    print(f"{df_X_rot_x_total_train.shape}")
    print(f"{df_X_rot_y_total_train.shape}")
    print(f"{df_X_rot_z_total_train.shape}")
    print(f"{df_y_train.shape}")

    print(f"{df_X_acc_x_total_test.shape}")
    print(f"{df_X_acc_y_total_test.shape}")
    print(f"{df_X_acc_z_total_test.shape}")
    print(f"{df_X_acc_x_gravity_test.shape}")
    print(f"{df_X_acc_y_gravity_test.shape}")
    print(f"{df_X_acc_z_gravity_test.shape}")
    print(f"{df_X_rot_x_total_test.shape}")
    print(f"{df_X_rot_y_total_test.shape}")
    print(f"{df_X_rot_z_total_test.shape}")
    print(f"{df_y_test.shape}")
print_dataset_size()
# ****************************************************************
# data augmentation 

cols = df_X_acc_x_total_train.columns
def add_jitter(data, sigma=0.01):
    # data shape: (num_windows, 128, 9)
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    return data + noise

def add_scaling(data, sigma=0.1):
    # Generate a random scaling factor for each window
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(data.shape[0], data.shape[1]))
    np.random.normal()
    return data * scaling_factor

def add_shift(data, shift_max=10):
    shifted_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        shift = np.random.randint(-shift_max, shift_max)
        np.roll
        shifted_data = pd.DataFrame(np.roll(data, shift, axis=1), columns=cols)
    return shifted_data

augment_data = True 
if augment_data:
    df_X_acc_x_total_jitter = add_jitter(df_X_acc_x_total_train)
    df_X_acc_y_total_jitter = add_jitter(df_X_acc_y_total_train)
    df_X_acc_z_total_jitter = add_jitter(df_X_acc_z_total_train)
    df_X_acc_x_gravity_jitter = add_jitter(df_X_acc_x_gravity_train)
    df_X_acc_y_gravity_jitter = add_jitter(df_X_acc_y_gravity_train)
    df_X_acc_z_gravity_jitter = add_jitter(df_X_acc_z_gravity_train)
    df_X_rot_x_total_jitter = add_jitter(df_X_rot_x_total_train)
    df_X_rot_y_total_jitter = add_jitter(df_X_rot_y_total_train)
    df_X_rot_z_total_jitter = add_jitter(df_X_rot_z_total_train)

    df_X_acc_x_total_scale = add_scaling(df_X_acc_x_total_train)
    df_X_acc_y_total_scale = add_scaling(df_X_acc_y_total_train)
    df_X_acc_z_total_scale = add_scaling(df_X_acc_z_total_train)
    df_X_acc_x_gravity_scale = add_scaling(df_X_acc_x_gravity_train)
    df_X_acc_y_gravity_scale = add_scaling(df_X_acc_y_gravity_train)
    df_X_acc_z_gravity_scale = add_scaling(df_X_acc_z_gravity_train)
    df_X_rot_x_total_scale = add_scaling(df_X_rot_x_total_train)
    df_X_rot_y_total_scale = add_scaling(df_X_rot_y_total_train)
    df_X_rot_z_total_scale = add_scaling(df_X_rot_z_total_train)

    df_X_acc_x_total_shift = add_shift(df_X_acc_x_total_train)
    df_X_acc_y_total_shift = add_shift(df_X_acc_y_total_train)
    df_X_acc_z_total_shift = add_shift(df_X_acc_z_total_train)
    df_X_acc_x_gravity_shift = add_shift(df_X_acc_x_gravity_train)
    df_X_acc_y_gravity_shift = add_shift(df_X_acc_y_gravity_train)
    df_X_acc_z_gravity_shift = add_shift(df_X_acc_z_gravity_train)
    df_X_rot_x_total_shift = add_shift(df_X_rot_x_total_train)
    df_X_rot_y_total_shift = add_shift(df_X_rot_y_total_train)
    df_X_rot_z_total_shift = add_shift(df_X_rot_z_total_train)

    # df_X_acc_x_total
    # df_X_acc_y_total
    # df_X_acc_z_total
    # df_X_acc_x_gravity
    # df_X_acc_y_gravity
    # df_X_acc_z_gravity
    # df_X_rot_x_total
    # df_X_rot_y_total
    # df_X_rot_z_total

    df_X_acc_x_total_train = pd.concat((df_X_acc_x_total_train, df_X_acc_x_total_jitter, df_X_acc_x_total_scale, df_X_acc_x_total_shift))
    df_X_acc_y_total_train = pd.concat((df_X_acc_y_total_train, df_X_acc_y_total_jitter, df_X_acc_y_total_scale, df_X_acc_y_total_shift))
    df_X_acc_z_total_train = pd.concat((df_X_acc_z_total_train, df_X_acc_z_total_jitter, df_X_acc_z_total_scale, df_X_acc_z_total_shift))
    df_X_acc_x_gravity_train = pd.concat((df_X_acc_x_gravity_train, df_X_acc_x_gravity_jitter, df_X_acc_x_gravity_scale, df_X_acc_x_gravity_shift))
    df_X_acc_y_gravity_train = pd.concat((df_X_acc_y_gravity_train, df_X_acc_y_gravity_jitter, df_X_acc_y_gravity_scale, df_X_acc_y_gravity_shift))
    df_X_acc_z_gravity_train = pd.concat((df_X_acc_z_gravity_train, df_X_acc_z_gravity_jitter, df_X_acc_z_gravity_scale, df_X_acc_z_gravity_shift))
    df_X_rot_x_total_train = pd.concat((df_X_rot_x_total_train, df_X_rot_x_total_jitter, df_X_rot_x_total_scale, df_X_rot_x_total_shift))
    df_X_rot_y_total_train = pd.concat((df_X_rot_y_total_train, df_X_rot_y_total_jitter, df_X_rot_y_total_scale, df_X_rot_y_total_shift))
    df_X_rot_z_total_train = pd.concat((df_X_rot_z_total_train, df_X_rot_z_total_jitter, df_X_rot_z_total_scale, df_X_rot_z_total_shift))
    df_y_train = pd.concat((df_y_train,df_y_train,df_y_train,df_y_train))

    print_dataset_size()
    
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

'''
x_acc_total, y_acc_total, z_acc_total, x_acc_gravity_y_acc_gravity, z_acc_gravity, x_rot_total, y_rot_total, z_rot_total 
'''
axes = [
    "acc_x_total", "acc_y_total", "acc_z_total",
    "acc_x_gravity", "acc_y_gravity", "acc_z_gravity",
    "rot_x_total", "rot_y_total", "rot_z_total"
]

def compute_features(data):
    # grab 9 axis from the list
    df_acc_x_total = data["acc_x_total"]
    df_acc_y_total = data["acc_y_total"]
    df_acc_z_total = data["acc_z_total"]
    df_acc_x_gravity = data["acc_x_gravity"]
    df_acc_y_gravity = data["acc_y_gravity"]
    df_acc_z_gravity = data["acc_z_gravity"]
    df_rot_x_total = data["rot_x_total"]
    df_rot_y_total = data["rot_y_total"]
    df_rot_z_total = data["rot_z_total"]
    
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
    # correlation 
    def get_correlation(a, b):
        return a.corrwith(b, axis=1)
    df_X_features["acc_x_y_total_corr"] = get_correlation(df_acc_x_total, df_acc_y_total)
    df_X_features["acc_y_z_total_corr"] = get_correlation(df_acc_y_total, df_acc_z_total)
    df_X_features["acc_x_z_total_corr"] = get_correlation(df_acc_z_total, df_acc_x_total)
    df_X_features["acc_x_y_gravity_corr"] = get_correlation(df_acc_x_gravity, df_acc_y_gravity)
    df_X_features["acc_y_z_gravity_corr"] = get_correlation(df_acc_y_gravity, df_acc_z_gravity)
    df_X_features["acc_x_z_gravity_corr"] = get_correlation(df_acc_z_gravity, df_acc_x_gravity)
    df_X_features["rot_x_y_total_corr"] = get_correlation(df_rot_x_total, df_rot_y_total)
    df_X_features["rot_y_z_total_corr"] = get_correlation(df_rot_y_total, df_rot_z_total)
    df_X_features["rot_x_z_total_corr"] = get_correlation(df_rot_z_total, df_rot_x_total)
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
    def compute_iqr_vectorized(windowed_data):
        """
        Computes IQR for a 2D array of windows.
        Input shape: (num_windows, 128)
        Output shape: (num_windows,)
        """
        # Calculate Q1 (25th percentile) and Q3 (75th percentile) along the samples axis
        q1 = np.percentile(windowed_data, 25, axis=1)
        q3 = np.percentile(windowed_data, 75, axis=1)
        
        return q3 - q1
    df_X_features["acc_x_total_iqr"] = compute_iqr_vectorized(df_acc_x_total)
    df_X_features["acc_y_total_iqr"] = compute_iqr_vectorized(df_acc_y_total)
    df_X_features["acc_z_total_iqr"] = compute_iqr_vectorized(df_acc_z_total)
    df_X_features["acc_x_gravity_iqr"] = compute_iqr_vectorized(df_acc_x_gravity)
    df_X_features["acc_y_gravity_iqr"] = compute_iqr_vectorized(df_acc_y_gravity)
    df_X_features["acc_z_gravity_iqr"] = compute_iqr_vectorized(df_acc_z_gravity)
    df_X_features["rot_x_total_iqr"] = compute_iqr_vectorized(df_rot_x_total)
    df_X_features["rot_y_total_iqr"] = compute_iqr_vectorized(df_rot_y_total)
    df_X_features["rot_z_total_iqr"] = compute_iqr_vectorized(df_rot_z_total)
    # entropy 
    def compute_entropy(ts_data):
        ts_data = ts_data.to_numpy().astype(np.float64)
        # 1. Perform FFT and get the magnitude (absolute value)
        fft_coeffs = np.fft.fft(ts_data, axis=1)
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
        windowed_data = windowed_data.to_numpy().astype(np.float64)
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
    print(f"X features size = {df_X_features.shape}")
    return df_X_features

# ****************************************************************
# The 9 component names (must match your file naming convention)
axes = [
    "acc_x_total", "acc_y_total", "acc_z_total",
    "acc_x_gravity", "acc_y_gravity", "acc_z_gravity",
    "rot_x_total", "rot_y_total", "rot_z_total"
]

time_series_train = [df_X_acc_x_total_train, 
              df_X_acc_y_total_train, 
              df_X_acc_z_total_train, 
              df_X_acc_x_gravity_train, 
              df_X_acc_y_gravity_train, 
              df_X_acc_z_gravity_train, 
              df_X_rot_x_total_train, 
              df_X_rot_y_total_train,
              df_X_rot_z_total_train]
time_series_test = [df_X_acc_x_total_test, 
              df_X_acc_y_total_test, 
              df_X_acc_z_total_test, 
              df_X_acc_x_gravity_test, 
              df_X_acc_y_gravity_test, 
              df_X_acc_z_gravity_test, 
              df_X_rot_x_total_test, 
              df_X_rot_y_total_test,
              df_X_rot_z_total_test]

time_series_train = {axis: df for axis, df in zip(axes, time_series_train)}
time_series_test = {axis: df for axis, df in zip(axes, time_series_test)}
df_X_features_train = compute_features(time_series_train)
df_X_features_test = compute_features(time_series_test)

# ****************************************************************
# normalize the time series data for the CNN

from sklearn.preprocessing import StandardScaler

# The 9 component names (must match your file naming convention)
axes = [
    "acc_x_total", "acc_y_total", "acc_z_total",
    "acc_x_gravity", "acc_y_gravity", "acc_z_gravity",
    "rot_x_total", "rot_y_total", "rot_z_total"
]

df_X_timeseries_train = [
    df_X_acc_x_total_train,
    df_X_acc_y_total_train,
    df_X_acc_z_total_train,
    df_X_acc_x_gravity_train,
    df_X_acc_y_gravity_train,
    df_X_acc_z_gravity_train,
    df_X_rot_x_total_train,
    df_X_rot_y_total_train,
    df_X_rot_z_total_train
]

# Create the dictionary
train_timeseries_dict = {axis: df for axis, df in zip(axes, df_X_timeseries_train)}

scalers = {}
scaled_dataframes = {}

'''
Exactly right. You’ve hit on a critical point for Time Series data that many people miss.

When you flatten the (num_windows, 128) dataframe into a single column of (num_windows * 128, 1) before fitting the StandardScaler, 
you are treating every single reading from that sensor as a sample from the same "global" distribution.
Why Global Scaling is Mandatory for CNNs

If you didn't flatten and instead scaled the (num_windows, 128) matrix directly, StandardScaler would treat it as 128 separate 
features. This would calculate 128 different means—one for "Sample #1 in the window," one for "Sample #2," and so on.

This would break your model for three reasons:

    Destroys the Waveform: If "Sample #1" is always the start of a pedal stroke and "Sample #60" is the peak, scaling them 
    with different means would "flatten" the wave, removing the very pattern your CNN is trying to learn.

    Temporal Invariance: In a CNN, a "step" or "jump" could start at Sample #5 or Sample #50. If each position in the window
    has its own scaling logic, the model can't recognize the same shape if it shifts in time.

    Physical Meaning: The physical property of the Accelerometer X-axis (e.g., gravity + linear motion) doesn't change 
    just because it's the 10th sample in a buffer. You want one μ (mean) and one σ (std) that represent that axis across 
    all time.
'''
for axis, df in train_timeseries_dict.items():
    scaler = StandardScaler()
    
    # 1. Convert to numpy and flatten to a 1D column for the scaler
    # We treat every single sample across all windows as a data point
    flat_data = df.to_numpy().reshape(-1, 1)
    
    # 2. Fit and Transform
    scaled_flat = scaler.fit_transform(flat_data)
    
    # 3. Reshape back to (num_windows, 128)
    scaled_df = scaled_flat.reshape(df.shape)
    scaled_df = pd.DataFrame(scaled_df, columns = range(window_size))
    scaled_dataframes[axis] = scaled_df
    scalers[axis] = scaler # Save for transforming test data later

    
# 1. Flatten your 3D training data to 2D to fit the scaler
# Shape: (N_train, 128, 12) -> (N_train * 128, 12)
# N, T, C = X_train.shape
# X_train_reshaped = X_train.reshape(-1, C)

# scaler = StandardScaler()
# X_train_normalized = scaler.fit_transform(df_X_timeseries)

# # 2. Reshape back to 3D
# X_train = X_train_normalized.reshape(N, T, C)

# # 3. Apply the SAME scaler to the Test set (DO NOT call fit_transform)
# N_t, T_t, C_t = X_test.shape
# X_test_reshaped = X_test.reshape(-1, C_t)
# X_test_normalized = scaler.transform(X_test_reshaped)
# X_test = X_test_normalized.reshape(N_t, T_t, C_t)

# ****************************************************************
# final dataset dir 
'''
final dataset file tree:

final_dataset/ 
    train/
        time_series/
            X_acc_x_total_train.csv 
            X_acc_y_total_train.csv
            X_acc_z_total_train.csv  
            X_acc_x_gravity_train.csv 
            X_acc_y_gravity_train.csv 
            X_acc_z_gravity_train.csv 
            X_rot_x_total_train.csv 
            X_rot_y_total_train.csv
            X_rot_z_total_train.csv  
        features/
            X_train.csv
        y_train.csv
    test/
        time_series/
            X_acc_x_total_test.csv 
            X_acc_y_total_test.csv
            X_acc_z_total_test.csv  
            X_acc_x_gravity_test.csv 
            X_acc_y_gravity_test.csv 
            X_acc_z_gravity_test.csv 
            X_rot_x_total_test.csv 
            X_rot_y_total_test.csv
            X_rot_z_total_test.csv  
        features/
            X_test.csv
        y_test.csv
    scalers/
        scalers.pkl
    activity_labels.csv
'''
from pathlib import Path

# Base Paths
base_path = Path("./final_dataset/")
train_path = base_path / "train"
test_path = base_path / "test"

# Define the 9 sensor components (axes)
axes = [
    "acc_x_total", "acc_y_total", "acc_z_total",
    "acc_x_gravity", "acc_y_gravity", "acc_z_gravity",
    "rot_x_total", "rot_y_total", "rot_z_total"
]

def get_paths(split):
    """
    Generates paths for a specific split ('train' or 'test').
    Returns a dictionary of time_series paths and the feature/label paths.
    """
    split_dir = base_path / split
    ts_dir = split_dir / "time_series"
    
    # 1. Map the 9 Time Series files
    ts_paths = {
        axis: ts_dir / f"X_{axis}_{split}.csv" for axis in axes
    }
    
    # 2. Map Features and Labels
    feature_path = split_dir / "features" / f"X_{split}.csv"
    label_path = split_dir / f"y_{split}.csv"
    
    return ts_paths, feature_path, label_path

# Generate for Train
train_ts_paths, x_train_features_path, y_train_path = get_paths("train")

# Generate for Test
test_ts_paths, x_test_features_path, y_test_path = get_paths("test")

# Activity Labels (Root level)
activity_labels_path = base_path / "activity_labels.csv"

# --- Quick Verification ---
print(f"Verified {len(train_ts_paths)} training time-series paths.")
print(f"Verified {len(test_ts_paths)} testing time-series paths.")
print(f"Example: {train_ts_paths['rot_z_total']}")

# ****************************************************************
# Define the base project directory
base_path = Path("./final_dataset")

# Define all the specific subdirectories you need
folders_to_create = [
    base_path / "train" / "time_series",
    base_path / "train" / "features",
    base_path / "test" / "time_series",
    base_path / "test" / "features",
    base_path / "scalers"
]
# Loop through and create them
for folder in folders_to_create:
    # parents=True: creates final_dataset and train if they don't exist
    # exist_ok=True: doesn't crash if the folder is already there
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Verified directory: {folder}")
print("\nFull folder structure is ready for data export.")
    
# Save Train Time Series
for axis_name, path in train_ts_paths.items():
    time_series_train[axis_name].to_csv(path, index=False)

# Save Test Time Series (assuming you have a similar test_timeseries_dict)
for axis_name, path in test_ts_paths.items():
    time_series_test[axis_name].to_csv(path, index=False)

# Save Hand-crafted Features
df_X_features_train.to_csv(x_train_features_path, index=False)
df_X_features_test.to_csv(x_test_features_path, index=False)

# Save Labels
df_y_train.to_csv(y_train_path, index=False)
df_y_test.to_csv(y_test_path, index=False)

# save scalers 
import joblib
scaler_filename = base_path / "scalers" / "9_axis_scalers.pkl"
try:
    joblib.dump(scalers, scaler_filename)
    print(f"Successfully saved 9 scalers to: {scaler_filename}")
except Exception as e:
    print(f"Error saving scalers: {e}")

pass
























































# ****************************************************************
# # Train-Test-Split 

# # perform a train-test-split on the indices 
# indices = np.arange(len(df_y))

# # 2. Split the indices, NOT the data yet
# # Stratify=y_labels ensures each activity is represented 80/20
# idx_train, idx_test = train_test_split(
#     indices, 
#     test_size=0.2, 
#     stratify=df_y, 
#     random_state=42
# )

# at this point the indices for train-test-split have been created. 
# print(idx_train, idx_test)

















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








# import numpy as np
# import pandas as pd

# def calculate_derivatives_and_mag(df):
#     """
#     Adds Jerk and Magnitude columns to your processed dataframe.
#     Assumes 50Hz (dt = 0.02s)
#     """
#     dt = 0.02
#     new_df = df.copy()
    
#     # 1. Calculate Magnitudes
#     for prefix in ['body_acc', 'grav_acc', 'clean_gyro']:
#         cols = [f'{prefix}_x', f'{prefix}_y', f'{prefix}_z']
#         new_df[f'{prefix}_mag'] = np.sqrt((df[cols]**2).sum(axis=1))
    
#     # 2. Calculate Jerk (Derivative of Body Acc and Gyro)
#     # We use np.gradient for a more stable central difference
#     for axis in ['x', 'y', 'z', 'mag']:
#         # Body Acc Jerk
#         new_df[f'body_acc_jerk_{axis}'] = np.gradient(new_df[f'body_acc_{axis}'], dt)
#         # Gyro Jerk (Angular Acceleration)
#         new_df[f'gyro_jerk_{axis}'] = np.gradient(new_df[f'clean_gyro_{axis}'], dt)
        
#     return new_df

# # Usage:
# # df_augmented = calculate_derivatives_and_mag(df_final)











# unique, counts = np.unique(y_test, return_counts=True)
# print("Test set distribution:", dict(zip(unique, counts)))



















# 1. The Golden Rule: Order of Operations

# The sequence should always be:

#     Denoise & Filter (Continuous Data)

#     Segment into Windows (Overlap 50%)

#     Stratified Train-Test Split (Indices)

#     Calculate Features (Optional: For Random Forest)

#     Normalize (Fit on Train, Transform Test)



































