from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

# ****************************************************************
# load training files, testing files, and scalers. 

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

# Scaler path 
scalers_path = base_path / "scalers" / "9_axis_scalers.pkl"

time_series_train = {}
time_series_test = {}
df_X_features_train = pd.DataFrame() 
df_X_features_test = pd.DataFrame()

# Load Train Time Series
for axis_name, path in train_ts_paths.items():
    time_series_train[axis_name] = pd.read_csv(path)

# Load Test Time Series (assuming you have a similar test_timeseries_dict)
for axis_name, path in test_ts_paths.items():
    time_series_test[axis_name] = pd.read_csv(path)

# Load Hand-crafted Features
df_X_features_train = pd.read_csv(x_train_features_path)
df_X_features_test = pd.read_csv(x_test_features_path)

# Load Labels
df_y_train = pd.read_csv(y_train_path)
df_y_test = pd.read_csv(y_test_path)

# Load back the registry
scalers = joblib.load(scalers_path)

# Example: Accessing the X-axis Accelerometer mean
print(f"Mean for Acc X: {scalers['acc_x_total'].mean_}")

def load_and_scale_cnn_data(X_dict, y, scalers_dict, transform=False):
    temp_stacked = []
    for axis, data in X_dict.items():
        # 1. Load the specific axis CSV
        data = data.to_numpy() # Shape: (num_windows, 128)
        
        # 2. Scale using the GLOBAL mean/std we saved
        # Flatten -> Transform -> Reshape back
        if transform == True:
            scaler = scalers_dict[axis]
            data_flat = data.reshape(-1, 1)
            scaled_data_flat = scaler.transform(data_flat)
            scaled_data = scaled_data_flat.reshape(data.shape)
        
            temp_stacked.append(scaled_data)
        else: 
            temp_stacked.append(data)
    # 3. Stack along the last axis (axis=2) to create 'channels'
    # Final Shape: (num_windows, 128, 9)
    X = np.stack(temp_stacked, axis=2)
    # Load labels
    y = y.to_numpy().ravel()
    return X, y

# Generate the 3D inputs
X_train_cnn, y_train_cnn = load_and_scale_cnn_data(time_series_train, df_y_train, scalers)
X_test_cnn, y_test_cnn = load_and_scale_cnn_data(time_series_test, df_y_test, scalers, transform=True)

# Initialize the classifier
# n_estimators=100 is a good start; random_state ensures reproducibility
rf_model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)

# Train the model
rf_model.fit(df_X_features_train, df_y_train.values.ravel())

# Make predictions
y_pred = rf_model.predict(df_X_features_test)

# Check results
print(f"Accuracy: {accuracy_score(df_y_test, y_pred):.4f}")
print(classification_report(df_y_test, y_pred))
# y_test are the true labels, y_pred are the model's predictions
cm = confusion_matrix(df_y_test, y_pred)
print(cm)


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

# C. Build and Train your 1D-CNN
model.fit(X_train_cnn, y_train_cnn, epochs=500, batch_size=16, verbose=1)

# D. Evaluate

loss, acc = model.evaluate(X_test_cnn, y_test_cnn, verbose=1)
print(loss)
print(acc)



# 1. Get raw probabilities: shape (num_windows, num_classes)
y_prob = model.predict(X_train_cnn)

# 2. Convert probabilities to class labels (index of the max value)
y_pred = np.argmax(y_prob, axis=1)

# y_test are the true labels, y_pred are the model's predictions
cm = confusion_matrix(y_train_cnn, y_pred)
print(cm)



# 1. Get raw probabilities: shape (num_windows, num_classes)
y_prob = model.predict(X_test_cnn)

# 2. Convert probabilities to class labels (index of the max value)
y_pred = np.argmax(y_prob, axis=1)

# y_test are the true labels, y_pred are the model's predictions
cm = confusion_matrix(y_test_cnn, y_pred)
print(cm)



pass
# def build_cnn_model(input_shape, num_classes):
#     model = Sequential([
#         # 1. First Convolutional Block
#         # Filters=32, Kernel=3 is standard for 50Hz signals
#         Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
#         MaxPooling1D(pool_size=2),
        
#         # 2. Second Convolutional Block
#         Conv1D(filters=64, kernel_size=3, activation='relu'),
#         MaxPooling1D(pool_size=2),
        
#         # 3. Flatten and Regularization
#         Flatten(),
#         Dropout(0.5), # Crucial because your dataset is small
        
#         # 4. Fully Connected Output
#         Dense(64, activation='relu'),
#         Dense(num_classes, activation='softmax') # Softmax for probability distribution
#     ])
    
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy', # Handles your 0-4 integer labels
#         metrics=['accuracy']
#     )
    
#     return model

# # Your input shape: (128 time steps, 12 channels)
# model = build_cnn_model((128, 12), 5)
# model.summary()

# import numpy as np
# import pandas as pd

# def load_dataset_to_3d(file_prefix='signal_', axes=None):
#     if axes is None:
#         axes = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
#     instance_list = []
    
#     for axis in axes:
#         # Load the 128-column CSV (header=None since we saved without headers)
#         df = pd.read_csv(f'{file_prefix}{axis}.csv', header=None)
#         # Convert to numpy and add a "depth" dimension
#         # Shape change: (N, 128) -> (N, 128, 1)
#         instance_list.append(df.values[:, :, np.newaxis])
    
#     # Stack along the last axis to get (N, 128, 6)
#     X = np.concatenate(instance_list, axis=-1)
    
#     # Load labels
#     y = pd.read_csv('y_labels.csv', header=None).values
    
#     return X, y

# # Execute
# X_train, y_train = load_dataset_to_3d()

# print(f"X_train shape: {X_train.shape}") # Expected: (Samples, 128, 6)
# print(f"y_train shape: {y_train.shape}") # Expected: (Samples, 1)




# 1. Initialize K-Fold (5 folds means 80/20 split each time)
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
    

#     fold_accuracies.append(acc)
#     print(f"Fold {fold+1} Accuracy: {acc:.4f}")

# # 3. Final Result
# print(f"\nFinal Cross-Validation Accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")


# # Load feature names
# # features = pd.read_csv('UCI HAR Dataset/features.txt', sep='\s+', header=None, names=['ID', 'Name'])

# # Load Training Data
# X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', sep='\s+', header=None)
# y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None)

# # Load Test Data
# X_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', sep='\s+', header=None)
# y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None)

# # Assign feature names to columns (optional but helpful for feature importance)
# X_train.columns = features['Name']
# X_test.columns = features['Name']




