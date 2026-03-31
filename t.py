import pandas as pd
from pathlib import Path 

data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}
df = pd.DataFrame(data)

# Add a new row using loc[]
df.loc[len(df)] = ["Charlie", 35]
print(df)

# Assuming df_resampled is your 50Hz synced dataframe 
dataset_dir = Path("./raw_dataset/")
experiment_name = "cycling_1"
df_resampled = pd.read_csv(Path(dataset_dir, f'{experiment_name}_combined_resampled.csv'))
print(df_resampled.head())

window_size = 128
start = 100
end = -1 if (start+window_size) > (df.shape[0]-1) else start+window_size 
print(f"[{start}, {end}]")
window = df_resampled.iloc[start:end]
print(window.head())

new_df = pd.DataFrame(columns=df_resampled.columns)
print(new_df.head())

print(new_df.head())