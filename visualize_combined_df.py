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
experiment_type = "downstairs"
experiment_index = 2
# train or test 
experiment_train_test = "test"

experiment_name = f"{experiment_type}_{experiment_index}"
dst_experiment_name = f"{experiment_name}_{experiment_train_test}"
window_size=128
destination_dir = Path("./selected_windows", dst_experiment_name)

# list of starting indices 
starting_indices = []

# read dataset of given experiment type and index.
try:
    df_resampled = pd.read_csv(Path(dataset_dir, f'{experiment_name}_combined_resampled.csv'))
    os.makedirs(destination_dir, exist_ok=True)
except Exception as e:
    print(f"exception occurred: {e}")
    exit()

# ****************************************************************
# filtering
from scipy.signal import medfilt, butter, filtfilt

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
    total_acc = filtfilt(b_noise, a_noise, median_filtered)
    
    # 3. Apply Low-Pass Butterworth Filter
    # 0.3 Hz default cutoff frequency
    cutoff=0.3
    b_grav, a_grav = create_lowpass_filter(cutoff, fs, 3)
    # axis = 1 - filter along rows
    gravity = filtfilt(b_grav, a_grav, total_acc)

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
    total_rot = filtfilt(b_noise, a_noise, median_filtered)

    # convert back to DataFrame 
    df_total_rot = pd.DataFrame(total_rot)
    
    return df_total_rot 


# filter the data 
df_resampled["acc_x_total"], df_resampled["acc_x_gravity"] = denoise_accl_signal(df_resampled["acc_x"])
df_resampled["acc_y_total"], df_resampled["acc_y_gravity"] = denoise_accl_signal(df_resampled["acc_y"])
df_resampled["acc_z_total"], df_resampled["acc_z_gravity"] = denoise_accl_signal(df_resampled["acc_z"])
df_resampled["rot_x_total"] = denoise_gyro_signal(df_resampled["rot_x"])
df_resampled["rot_y_total"] = denoise_gyro_signal(df_resampled["rot_y"])
df_resampled["rot_z_total"] = denoise_gyro_signal(df_resampled["rot_z"])

# at this point the data has been filtered. 

# current starting point of the sliding window
cur_start = 0

'''
plotting function 
'''
def plot_interactive_windows(df: pd.DataFrame):
    global cur_start
    
    # Setup the figure and subplots (Accel and Gyro)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    plt.subplots_adjust(bottom=0.25) # Make room for the slider

    # Initial data slice
    cur_start = 0
    window = df.iloc[cur_start : cur_start + window_size]
    t = np.arange(window_size)

    # Plot Accelerometer
    line_ax, = ax1.plot(t, window['acc_x_total'], label='Acc X')
    line_ay, = ax1.plot(t, window['acc_y_total'], label='Acc Y')
    line_az, = ax1.plot(t, window['acc_z_total'], label='Acc Z')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.legend(loc='upper right')
    ax1.set_title(f'Window View (Size: {window_size})')

    #  plot gravity 
    line_ax_g, = ax2.plot(t, window['acc_x_gravity'], label='Acc X G')
    line_ay_g, = ax2.plot(t, window['acc_y_gravity'], label='Acc Y G')
    line_az_g, = ax2.plot(t, window['acc_z_gravity'], label='Acc Z G')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_xlabel('Sample Index within Window')
    ax2.legend(loc='upper right')
    
    # Plot Gyroscope
    line_gx, = ax3.plot(t, window['rot_x_total'], label='Rot X')
    line_gy, = ax3.plot(t, window['rot_y_total'], label='Rot Y')
    line_gz, = ax3.plot(t, window['rot_z_total'], label='Rot Z')
    ax3.set_ylabel('Angular Velocity (rad/s)')
    ax3.set_xlabel('Sample Index within Window')
    ax3.legend(loc='upper right')
    
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
        line_ax.set_ydata(new_window['acc_x_total'])
        line_ay.set_ydata(new_window['acc_y_total'])
        line_az.set_ydata(new_window['acc_z_total'])

        line_ax_g.set_ydata(new_window['acc_x_gravity'])
        line_ay_g.set_ydata(new_window['acc_y_gravity'])
        line_az_g.set_ydata(new_window['acc_z_gravity'])
        
        # Update Gyroscope lines
        line_gx.set_ydata(new_window['rot_x_total'])
        line_gy.set_ydata(new_window['rot_y_total'])
        line_gz.set_ydata(new_window['rot_z_total'])
        
        # Rescale y-axes automatically to fit new data
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax3.relim()
        ax3.autoscale_view()
        
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
        # example filename: cycling_1_acc_x_total.csv
        df_output_acc_x_total = pd.DataFrame(columns=range(128))
        df_output_acc_y_total = pd.DataFrame(columns=range(128))
        df_output_acc_z_total = pd.DataFrame(columns=range(128))
        df_output_acc_x_gravity = pd.DataFrame(columns=range(128))
        df_output_acc_y_gravity = pd.DataFrame(columns=range(128))
        df_output_acc_z_gravity = pd.DataFrame(columns=range(128))
        df_output_rot_x_total = pd.DataFrame(columns=range(128))
        df_output_rot_y_total = pd.DataFrame(columns=range(128))
        df_output_rot_z_total = pd.DataFrame(columns=range(128))
            
        # for every starting index in the list 
        for start in starting_indices:
            count_text.set_text(f"{start}") 
            # slice the data 
            end = -1 if (start+window_size) > (df.shape[0]-1) else start+window_size 
            print(f"[{start}, {end}]")
            window = df.iloc[start:end]

            # concatenate sliced data to output dfs
            cur_acc_x_total = window["acc_x_total"].values
            cur_acc_y_total = window["acc_y_total"].values
            cur_acc_z_total = window["acc_z_total"].values
            cur_acc_x_gravity = window["acc_x_gravity"].values
            cur_acc_y_gravity = window["acc_y_gravity"].values
            cur_acc_z_gravity = window["acc_z_gravity"].values
            cur_rot_x_total = window["rot_x_total"].values
            cur_rot_y_total = window["rot_y_total"].values
            cur_rot_z_total = window["rot_z_total"].values
            
            df_output_acc_x_total.loc[len(df_output_acc_x_total)] = cur_acc_x_total
            df_output_acc_y_total.loc[len(df_output_acc_y_total)] = cur_acc_y_total
            df_output_acc_z_total.loc[len(df_output_acc_z_total)] = cur_acc_z_total
            df_output_acc_x_gravity.loc[len(df_output_acc_x_gravity)] = cur_acc_x_gravity
            df_output_acc_y_gravity.loc[len(df_output_acc_y_gravity)] = cur_acc_y_gravity
            df_output_acc_z_gravity.loc[len(df_output_acc_z_gravity)] = cur_acc_z_gravity
            df_output_rot_x_total.loc[len(df_output_rot_x_total)] = cur_rot_x_total
            df_output_rot_y_total.loc[len(df_output_rot_y_total)] = cur_rot_y_total
            df_output_rot_z_total.loc[len(df_output_rot_z_total)] = cur_rot_z_total
        
        # write the six axis CSV files 
        # output filepath
        df_output_acc_x_total_ts_filepath = Path(destination_dir, f"{experiment_name}_acc_x_total_{experiment_train_test}.csv")
        df_output_acc_y_total_ts_filepath = Path(destination_dir, f"{experiment_name}_acc_y_total_{experiment_train_test}.csv")
        df_output_acc_z_total_ts_filepath = Path(destination_dir, f"{experiment_name}_acc_z_total_{experiment_train_test}.csv")
        df_output_acc_x_gravity_ts_filepath = Path(destination_dir, f"{experiment_name}_acc_x_gravity_{experiment_train_test}.csv")
        df_output_acc_y_gravity_ts_filepath = Path(destination_dir, f"{experiment_name}_acc_y_gravity_{experiment_train_test}.csv")
        df_output_acc_z_gravity_ts_filepath = Path(destination_dir, f"{experiment_name}_acc_z_gravity_{experiment_train_test}.csv")
        df_output_rot_x_total_ts_filepath = Path(destination_dir, f"{experiment_name}_rot_x_total_{experiment_train_test}.csv")
        df_output_rot_y_total_ts_filepath = Path(destination_dir, f"{experiment_name}_rot_y_total_{experiment_train_test}.csv")
        df_output_rot_z_total_ts_filepath = Path(destination_dir, f"{experiment_name}_rot_z_total_{experiment_train_test}.csv")
        
        df_output_acc_x_total.to_csv(df_output_acc_x_total_ts_filepath)
        df_output_acc_y_total.to_csv(df_output_acc_y_total_ts_filepath)
        df_output_acc_z_total.to_csv(df_output_acc_z_total_ts_filepath)
        df_output_acc_x_gravity.to_csv(df_output_acc_x_gravity_ts_filepath)
        df_output_acc_y_gravity.to_csv(df_output_acc_y_gravity_ts_filepath)
        df_output_acc_z_gravity.to_csv(df_output_acc_z_gravity_ts_filepath)
        df_output_rot_x_total.to_csv(df_output_rot_x_total_ts_filepath)
        df_output_rot_y_total.to_csv(df_output_rot_y_total_ts_filepath)
        df_output_rot_z_total.to_csv(df_output_rot_z_total_ts_filepath)
       
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

