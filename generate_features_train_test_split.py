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

print(f"{df_X_acc_x.shape}")
print(f"{df_X_acc_y.shape}")
print(f"{df_X_acc_z.shape}")
print(f"{df_X_rot_x.shape}")
print(f"{df_X_rot_y.shape}")
print(f"{df_X_rot_z.shape}")
print(f"{df_y.shape}")