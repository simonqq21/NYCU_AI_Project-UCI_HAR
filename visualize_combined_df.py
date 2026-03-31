import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path 

# Assuming df_resampled is your 50Hz synced dataframe 
dataset_dir = Path("./raw_dataset/")

# Set experiment type and index 
experiment_type = "cycling"
experiment_index = 1
experiment_name = f"{experiment_type}_{experiment_index}"
window_size=128

# list of starting indices 
starting_indices = []

# read dataset of given experiment type and index.
df_resampled = pd.read_csv(Path(dataset_dir, f'{experiment_name}_combined_resampled.csv'))


            
# functions to compute for features 



    
def plot_interactive_windows(df: pd.DataFrame):
    # Setup the figure and subplots (Accel and Gyro)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(bottom=0.25) # Make room for the slider

    # Initial data slice
    initial_start = 0
    window = df.iloc[initial_start : initial_start + window_size]
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
        start = int(val)
        new_window = df.iloc[start : start + window_size]
        
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

    # count textbox 
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

        '''
        final_dataset/ 
            train /
                time_series/
                features/
                
            test/
            activity_labels.csv
        '''
        
        # write the six axis CSV files 
        # output filepath
        cur_acc_x_time_series_filepath = Path(dataset_dir, f"{experiment_name}_starting_indices.csv")
        output_df.to_csv(output_filepath)

       
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