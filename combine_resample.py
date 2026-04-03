import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import csv  
from pathlib import Path 
import re

os.getcwd()

def get_all_raw_ds_folder_paths(dataset_dir):
    # all folders in the raw dataset folder
    folders = [f.name for f in dataset_dir.iterdir() if f.is_dir()]
    # eg. cycling_1, running_2, jumprope_1, ...
    pattern = r'.+_\d+'
    folder_results = sorted([f for f in folders if re.search(pattern, f)])
    print(folder_results)
    folder_results = [Path(dataset_dir, folder) for folder in folder_results]
    print(folder_results)
    return folder_results

dataset_dir = Path("./raw_dataset/")
raw_ds_folders = get_all_raw_ds_folder_paths(dataset_dir)
raw_ds_folders

final_dataset_folder = "./final_dataset" 
os.makedirs(final_dataset_folder, exist_ok=True)

def read_sensor_data_from_raw_ds_folder(raw_ds_folder):
    # 1. Load your data (assuming CSVs from Phyphox)
    # read timestamp, accelerometer, gyroscope, and pressure  
    df_accl = pd.read_csv(Path(raw_ds_folder, "Accelerometer.csv"))
    print(f"accelerometer dataframe shape: f{df_accl.shape}")
    df_gyro = pd.read_csv(Path(raw_ds_folder, "Gyroscope.csv"))
    print(f"gyroscope dataframe shape: f{df_gyro.shape}")
    df_pres = pd.read_csv(Path(raw_ds_folder, "Pressure.csv"))
    print(f"pressure dataframe shape: f{df_pres.shape}")
    # accelerometer, gyroscope, pressure 
    return df_accl, df_gyro, df_pres

def combine_resample_dfs(df_accl = None, df_gyro = None, df_pres = None):
    # Let's assume 'time' is the column name from Phyphox
    # Convert the 'time' column to a Timedelta or Datetime index 
    df_accl.columns = ["time", "acc_x", "acc_y", "acc_z"]
    df_accl["time"] = pd.to_timedelta(df_accl["time"], unit='s')
    df_accl.set_index('time', inplace=True)

    df_gyro.columns = ["time", "rot_x", "rot_y", "rot_z"]
    df_gyro["time"] = pd.to_timedelta(df_gyro["time"], unit='s')
    df_gyro.set_index('time', inplace=True)

    # Combine accelerometer and gyroscope rows with outer join
    # 2. Merge (Sync) the sensors
    # 'outer' join ensures we don't lose data from either sensor
    df_combined = df_accl.join(df_gyro, how="outer") 

    # 3. Interpolate and Resample to 50Hz (20ms intervals)
    # First, fill the NaNs created by the slight misalignment
    df_combined = df_combined.interpolate(method='linear')
    # Now, resample to exactly 50Hz
    # '20L' stands for 20 milliseconds
    df_resampled = df_combined.resample('20ms').mean()
    
    # Drop any remaining NaNs at the very start/end
    df_resampled.dropna(inplace=True)
    df_resampled.shape
    return df_resampled

def save_combined_resampled_df(df_resampled, raw_ds_folder):
    dst_filepath = f"{raw_ds_folder}_combined_resampled.csv" 
    df_resampled.to_csv(dst_filepath)

raw_ds_folder = raw_ds_folders[0]
df_accl, df_gyro, df_pres = read_sensor_data_from_raw_ds_folder(raw_ds_folder)
df_resampled = combine_resample_dfs(df_accl, df_gyro)
df_resampled.info()
save_combined_resampled_df(df_resampled, raw_ds_folder)

df_resampled.head()

print(raw_ds_folders)

for raw_ds_folder_path in raw_ds_folders:
    df_accl, df_gyro, df_pres = read_sensor_data_from_raw_ds_folder(raw_ds_folder_path)
    df_resampled = combine_resample_dfs(df_accl, df_gyro)
    df_resampled.info()
    save_combined_resampled_df(df_resampled, raw_ds_folder_path)

os.listdir("raw_dataset/")

