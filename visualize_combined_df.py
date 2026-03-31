import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path 
import os 

# Assuming df_resampled is your 50Hz synced dataframe 
dataset_dir = Path("./raw_dataset/")
# Set experiment type and index 
# cycling, running, jumprope, upstairs, downstairs
experiment_type = "upstairs"
experiment_index = 2

experiment_name = f"{experiment_type}_{experiment_index}"
window_size=128
destination_dir = Path("./selected_windows", experiment_name)

# list of starting indices 
starting_indices = []

# read dataset of given experiment type and index.
try:
    df_resampled = pd.read_csv(Path(dataset_dir, f'{experiment_name}_combined_resampled.csv'))
    os.makedirs(destination_dir, exist_ok=True)
except Exception as e:
    print(f"exception occurred: {e}")
    exit()

# current starting point of the sliding window
cur_start = 0

'''
plotting function 
'''
def plot_interactive_windows(df: pd.DataFrame):
    global cur_start
    
    # Setup the figure and subplots (Accel and Gyro)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(bottom=0.25) # Make room for the slider

    # Initial data slice
    cur_start = 0
    window = df.iloc[cur_start : cur_start + window_size]
    t = np.arange(window_size)

    # Plot Accelerometer
    line_ax, = ax1.plot(t, window['acc_x'], label='Acc X')
    line_ay, = ax1.plot(t, window['acc_y'], label='Acc Y')
    line_az, = ax1.plot(t, window['acc_z'], label='Acc Z')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.legend(loc='upper right')
    ax1.set_title(f'Window View (Size: {window_size})')

    # Plot Gyroscope
    line_gx, = ax2.plot(t, window['rot_x'], label='Rot X')
    line_gy, = ax2.plot(t, window['rot_y'], label='Rot Y')
    line_gz, = ax2.plot(t, window['rot_z'], label='Rot Z')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_xlabel('Sample Index within Window')
    ax2.legend(loc='upper right')
    
    # Add the Slider
    ax_slider = plt.axes([0.1, 0.1, 0.6, 0.03]) # [left, bottom, width, height]
    slider = Slider(
        ax_slider, 'Start Index', 0, len(df) - window_size, 
        valinit=2000, valstep=1
    )

    # Update function called when slider moves
    def update(val):
        global cur_start
        cur_start = int(val)
        new_window = df.iloc[cur_start : cur_start + window_size]
        
        # Update Accelerometer lines
        line_ax.set_ydata(new_window['acc_x'])
        line_ay.set_ydata(new_window['acc_y'])
        line_az.set_ydata(new_window['acc_z'])
        
        # Update Gyroscope lines
        line_gx.set_ydata(new_window['rot_x'])
        line_gy.set_ydata(new_window['rot_y'])
        line_gz.set_ydata(new_window['rot_z'])
        
        # Rescale y-axes automatically to fit new data
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # sample count textbox 
    count_text_ax = plt.axes([0.91,0.09,0.03,0.04],)
    count_text_ax.set_xticks([]) 
    count_text_ax.set_yticks([])
    count_text_ax.axis("off")
    count_text = count_text_ax.text(0.5,0.5,f"{len(starting_indices)}", 
                                    fontsize=16,)
    
    # clear button 
    # clears the index list 
    clear_btn_ax = plt.axes([0.75, 0.1, 0.05, 0.04])
    clear_btn = Button(clear_btn_ax, 'Clear')
    def reset(event):
        # clear all starting indices
        starting_indices.clear()
        # update counter text
        count_text.set_text(f"{len(starting_indices)}")
        # render UI
        fig.canvas.draw_idle()
    clear_btn.on_clicked(reset)

    # +half window button 
    half_btn_ax = plt.axes([0.75,0.06, 0.05, 0.04])
    half_btn = Button(half_btn_ax, "+half")
    def add_half(event):
        global cur_start
        cur_start = cur_start + window_size//2
        slider.set_val(cur_start)
        update(cur_start)
    half_btn.on_clicked(add_half)
    
    # save button 
    # appends the slider start index to the end of the list
    save_btn_ax = plt.axes([0.80, 0.1, 0.05, 0.04])
    save_btn = Button(save_btn_ax, 'Save')
    def add(event): 
        # append current slider value to starting indices
        starting_indices.append(slider.val)
        # update counter text
        count_text.set_text(f"{len(starting_indices)}") 
        # render UI
        fig.canvas.draw_idle()
    save_btn.on_clicked(add)

    # function to save time series CSV 
    def write_csvs():
        # six CSV files per activity
        # each CSV files has N rows and <window_size> columns
        # acc xyz, rot xyz
        # {activity}_{activity_number}_{sensor}_{axis}.csv
        # example filename: cycling_1_acc_x.csv
        df_output_acc_x = pd.DataFrame(columns=range(128))
        df_output_acc_y = pd.DataFrame(columns=range(128))
        df_output_acc_z = pd.DataFrame(columns=range(128))
        df_output_rot_x = pd.DataFrame(columns=range(128))
        df_output_rot_y = pd.DataFrame(columns=range(128))
        df_output_rot_z = pd.DataFrame(columns=range(128))
            
        # for every starting index in the list 
        for start in starting_indices:
            count_text.set_text(f"{start}") 
            # slice the data 
            end = -1 if (start+window_size) > (df.shape[0]-1) else start+window_size 
            print(f"[{start}, {end}]")
            window = df.iloc[start:end]

            # concatenate sliced data to output dfs
            cur_acc_x = window["acc_x"].values
            cur_acc_y = window["acc_y"].values
            cur_acc_z = window["acc_z"].values
            cur_rot_x = window["rot_x"].values
            cur_rot_y = window["rot_y"].values
            cur_rot_z = window["rot_z"].values
            
            df_output_acc_x.loc[len(df_output_acc_x)] = cur_acc_x
            df_output_acc_y.loc[len(df_output_acc_y)] = cur_acc_y
            df_output_acc_z.loc[len(df_output_acc_z)] = cur_acc_z
            df_output_rot_x.loc[len(df_output_rot_x)] = cur_rot_x
            df_output_rot_y.loc[len(df_output_rot_y)] = cur_rot_y
            df_output_rot_z.loc[len(df_output_rot_z)] = cur_rot_z
        
        # write the six axis CSV files 
        # output filepath
        df_output_acc_x_ts_filepath = Path(destination_dir, f"{experiment_name}_acc_x.csv")
        df_output_acc_y_ts_filepath = Path(destination_dir, f"{experiment_name}_acc_y.csv")
        df_output_acc_z_ts_filepath = Path(destination_dir, f"{experiment_name}_acc_z.csv")
        df_output_rot_x_ts_filepath = Path(destination_dir, f"{experiment_name}_rot_x.csv")
        df_output_rot_y_ts_filepath = Path(destination_dir, f"{experiment_name}_rot_y.csv")
        df_output_rot_z_ts_filepath = Path(destination_dir, f"{experiment_name}_rot_z.csv")
        
        df_output_acc_x.to_csv(df_output_acc_x_ts_filepath)
        df_output_acc_y.to_csv(df_output_acc_y_ts_filepath)
        df_output_acc_z.to_csv(df_output_acc_z_ts_filepath)
        df_output_rot_x.to_csv(df_output_rot_x_ts_filepath)
        df_output_rot_y.to_csv(df_output_rot_y_ts_filepath)
        df_output_rot_z.to_csv(df_output_rot_z_ts_filepath)
       
    # write button 
    # writes the list into a CSV file 
    write_btn_ax = plt.axes([0.85, 0.1, 0.05, 0.04])
    write_btn = Button(write_btn_ax, 'Write', color="green")
    def write(event):
        write_csvs()
        
        
 
    write_btn.on_clicked(write)
    
    plt.show()

# Run the visualizer
plot_interactive_windows(df_resampled)


























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


'''
denoise and filter accelerometer signals 
median filter for initial filtering
20 Hz lowpass butterworth filter for high frequency noise
0.3 Hz lowpass butterworth filter for gravity
'''
# accelerometer signal denoises and filters out 
def denoise_accl_signal(data, fs=50, order=3):
    nyq = 0.5 * fs

    # 1. Apply Median Filter (kernel size 3 or 5 is standard)
    median_filtered = medfilt(data, kernel_size=3)

    # 2. Apply Low-Pass Butterworth Filter
    # 20 Hz default cutoff frequency
    cutoff=20
    low = cutoff / nyq
    b_noise, a_noise = butter(order, low, btype='low')
    total_acc = filtfilt(b_noise, a_noise, median_filtered)
    
    # 3. Apply Low-Pass Butterworth Filter
    # 0.3 Hz default cutoff frequency
    cutoff=0.3
    low = cutoff / nyq
    b_grav, a_grav = butter(order, low, btype='low')
    gravity = filtfilt(b_grav, a_grav, total_acc)

    body_acc = total_acc - gravity

    return body_acc, gravity 

'''
denoise and filter gyroscope signals 
median filter for initial filtering
20 Hz lowpass butterworth filter for high frequency noise
'''
def denoise_gyro_signal(data, fs=50, order=3):
    nyq = 0.5 * fs

    # 1. Apply Median Filter (kernel size 3 or 5 is standard)
    median_filtered = medfilt(data, kernel_size=3)

    # 2. Apply Low-Pass Butterworth Filter
    # 20 Hz default cutoff frequency
    cutoff=20
    low = cutoff / nyq
    b_noise, a_noise = butter(order, low, btype='low')
    total_rot = filtfilt(b_noise, a_noise, median_filtered)
    
    return total_rot