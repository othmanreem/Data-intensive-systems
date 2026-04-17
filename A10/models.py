import os
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Load Kinect movement data

JOINTS = ["head", "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
          "left_hand", "right_hand", "left_hip", "right_hip", "left_knee", "right_knee",
          "left_foot", "right_foot",
]
N_JOINTS = len(JOINTS)          # total number of body joints
N_INPUT  = N_JOINTS * 2         # input for each joint(x and y)
N_OUTPUT = N_JOINTS * 1         # output/target z coordiante
print(f'Input: {N_INPUT} \tOutput:{N_OUTPUT}')
print(f'Joints: {N_JOINTS}')

# Loads single csv file and splits into input and target array
def load_single_csv(filepath_or_bytes):
    if isinstance(filepath_or_bytes, (str, os.PathLike)):
        df = pd.read_csv(filepath_or_bytes)
    else:
        df = pd.read_csv(io.BytesIO(filepath_or_bytes))
    df.columns = df.columns.str.strip()

    x_cols = [f"{j}_x" for j in JOINTS]
    y_cols = [f"{j}_y" for j in JOINTS]
    z_cols = [f"{j}_z" for j in JOINTS]

    xy_cols = []
    for j in JOINTS:
        xy_cols += [f"{j}_x", f"{j}_y"]

    X = df[xy_cols].values.astype(np.float32)  # input
    y = df[z_cols].values.astype(np.float32)    # Target

    return X, y

# load all csv file from the folder
def load_all_sequences(folder_path):
    sequences, file_names = [], []

    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    csv_files.sort()

    print(f"Found {len(csv_files)} CSV files in folder.")

    for name in csv_files:
        file_path = os.path.join(folder_path, name)

        with open(file_path, 'rb') as f:
            raw = f.read()

        X, y = load_single_csv(raw)

        sequences.append((X, y))   # stores as (input,target)
        file_names.append(name)

    return sequences, file_names

# For Dense MLP model, which treats each frame independently
def flatten_sequences(sequences):
    X_flat = np.concatenate([s[0] for s in sequences], axis=0)
    y_flat = np.concatenate([s[1] for s in sequences], axis=0)
    return X_flat, y_flat

# Create fixed-length windows of consecutive frames from each session for conv1d, lstm and gru
def make_windowed_sequences(sequences, window_size=30, stride=1):
    X_list, y_list = [], []
    for X, y in sequences:
        n = len(X)
        for start in range(0, n - window_size + 1, stride):
            X_list.append(X[start : start + window_size])
            y_list.append(y[start : start + window_size])

    X_seq = np.array(X_list, dtype=np.float32)   # (N, window, 26)
    y_seq = np.array(y_list, dtype=np.float32)   # (N, window, 13)
    return X_seq, y_seq

# Split sequences by session first, then window within each split
# This prevents data leakage by ensuring no temporal overlap between train/test
def split_and_window_sequences(sequences, file_names, test_size=0.2, window_size=30, stride=5, random_state=42):
    """
    Split sequences into train and test sets BEFORE windowing to prevent data leakage.

    The key fix: Split by session first, then window within each split.
    This ensures that windows from the same session don't appear in both train and test sets.

    Args:
        sequences: List of (X, y) tuples, one per session
        file_names: List of filenames corresponding to each sequence
        test_size: Proportion of sessions to use for testing
        window_size: Size of each window
        stride: Stride between windows
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, train_file_names, test_file_names)
        where y_train and y_test contain both windowed targets and last-frame targets
    """
    from sklearn.model_selection import train_test_split

    # Split sessions (not frames!) into train and test
    # Use indices to ensure file_names stay aligned with sequences
    all_indices = list(range(len(sequences)))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    train_sessions = [sequences[i] for i in train_indices]
    test_sessions = [sequences[i] for i in test_indices]
    train_files = [file_names[i] for i in train_indices]
    test_files = [file_names[i] for i in test_indices]

    print(f"Training sessions: {len(train_sessions)}, Test sessions: {len(test_sessions)}")

    # Now window each split separately
    X_train_seq, y_train_seq = make_windowed_sequences(train_sessions, window_size=window_size, stride=stride)
    X_test_seq, y_test_seq = make_windowed_sequences(test_sessions, window_size=window_size, stride=stride)

    # Also create last-frame targets for convenience
    y_train_last = y_train_seq[:, -1, :]  # (N_train, 13)
    y_test_last = y_test_seq[:, -1, :]    # (N_test, 13)

    print(f"Training windows: {len(X_train_seq)}")
    print(f"Test windows: {len(X_test_seq)}")

    return (
        X_train_seq, y_train_seq, y_train_last,
        X_test_seq, y_test_seq, y_test_last,
        train_files, test_files
    )


# Frame-level split (for Dense models) - also split by session first
def split_flat_sequences(sequences, file_names, test_size=0.2, random_state=42):
    """
    Split sequences into train and test sets for flat (frame-level) models.
    Also splits by session first to prevent leakage.
    """
    from sklearn.model_selection import train_test_split

    # Split sessions
    # Split sessions (not frames!) into train and test
    # Each element is a session (X, y) tuple
    # Use indices to ensure file_names stay aligned with sequences
    all_indices = list(range(len(sequences)))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    train_sessions = [sequences[i] for i in train_indices]
    test_sessions = [sequences[i] for i in test_indices]
    train_files = [file_names[i] for i in train_indices]
    test_files = [file_names[i] for i in test_indices]

    # Then flatten within each split
    X_train_flat, y_train_flat = flatten_sequences(train_sessions)
    X_test_flat, y_test_flat = flatten_sequences(test_sessions)

    print(f"Training frames: {len(X_train_flat)}, Test frames: {len(X_test_flat)}")

    return X_train_flat, y_train_flat, X_test_flat, y_test_flat, train_files, test_files

REPO_ROOT    = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_DIR     = os.path.join(REPO_ROOT, 'Datasets_all')
KINECT_DATA_PATH = os.path.join(DATA_DIR, 'kinect_good_preprocessed')

# sequences contain list of tuples (X,y)
sequences, file_names = load_all_sequences(KINECT_DATA_PATH)

print(f"\nTotal sessions loaded: {len(sequences)}")

# Frame-level flat data  (for Dense models) - NOW WITH PROPER SPLIT
X_flat_train, y_flat_train, X_flat_test, y_flat_test, train_files, test_files = split_flat_sequences(
    sequences, file_names, test_size=0.2, random_state=42
)
print(f"\nFlat dataset - Train: X={X_flat_train.shape}  y={y_flat_train.shape}")
print(f"Flat dataset - Test:  X={X_flat_test.shape}  y={y_flat_test.shape}")
print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

# Windowed sequences  (for Conv1D / LSTM / GRU models) - NOW WITH PROPER SPLIT
WINDOW_SIZE = 30
(
    X_seq_train, y_seq_train, y_train_last,
    X_seq_test, y_seq_test, y_test_last,
    train_files_win, test_files_win
) = split_and_window_sequences(
    sequences, file_names,
    test_size=0.2,
    window_size=WINDOW_SIZE,
    stride=5,
    random_state=42
)
print(f"\nWindowed dataset - Train: X={X_seq_train.shape}  y_last={y_train_last.shape}")
print(f"Windowed dataset - Test:  X={X_seq_test.shape}  y_last={y_test_last.shape}")
print(f"Train files: {len(train_files_win)}, Test files: {len(test_files_win)}")

# Define Deep Learning network architectures

# DEnse MLP

def build_dense_model( hidden_units=(128, 64), activation="relu", dropout_rate=0.2,
                       l2_reg=1e-4,optimizer="adam", loss="mse",
):
    inputs = keras.Input(shape=(N_INPUT,), name="xy_input")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None,
            name=f"dense_{i+1}",
        )(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    outputs = layers.Dense(N_OUTPUT, activation="linear", name="z_output")(x)
    model = keras.Model(inputs, outputs, name="DenseModel")
    return model

# Conv1D CNN

def build_conv1d_model(filters=(64, 128), kernel_size=3, pool_size=2, dense_units=(64,),
                       activation="relu", dropout_rate=0.2,optimizer="adam", loss="mse",
):
    inputs = keras.Input(shape=(WINDOW_SIZE, N_INPUT), name="xy_seq_input")
    x = inputs
    for i, f in enumerate(filters):
        x = layers.Conv1D(f, kernel_size, activation=activation, padding="same",
                          name=f"conv_{i+1}")(x)
        x = layers.MaxPooling1D(pool_size, padding="same", name=f"pool_{i+1}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"drop_conv_{i+1}")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation=activation, name=f"fc_{i+1}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"drop_fc_{i+1}")(x)

    outputs = layers.Dense(N_OUTPUT, activation="linear", name="z_output")(x)
    model = keras.Model(inputs, outputs, name="Conv1DModel")
    return model

# layers.LSTM

def build_lstm_model(lstm_units=(64, 32), dense_units=(32,), activation="tanh",
                     dropout_rate=0.2, recurrent_dropout=0.0, optimizer="adam", loss="mse",
):
    inputs = keras.Input(shape=(WINDOW_SIZE, N_INPUT), name="xy_seq_input")
    x = inputs
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            name=f"lstm_{i+1}",
        )(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name=f"fc_{i+1}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"drop_fc_{i+1}")(x)

    outputs = layers.Dense(N_OUTPUT, activation="linear", name="z_output")(x)
    model = keras.Model(inputs, outputs, name="LSTMModel")
    return model

# layers.GRU

def build_gru_model(gru_units=(64, 32), dense_units=(32,), dropout_rate=0.2,
                    recurrent_dropout=0.0, optimizer="adam", loss="mse",):
    inputs = keras.Input(shape=(WINDOW_SIZE, N_INPUT), name="xy_seq_input")
    x = inputs
    for i, units in enumerate(gru_units):
        return_sequences = (i < len(gru_units) - 1)
        x = layers.GRU(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            name=f"gru_{i+1}",
        )(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name=f"fc_{i+1}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"drop_fc_{i+1}")(x)

    outputs = layers.Dense(N_OUTPUT, activation="linear", name="z_output")(x)
    model = keras.Model(inputs, outputs, name="GRUModel")
    return model

"""

## Key Changes Made to Fix Data Leakage

### 1. **New `split_and_window_sequences()` function** (lines 107-161)
- **Splits sessions first**: Uses `train_test_split()` on the list of sessions (not individual frames)
- **Then windows within each split**: Creates windows only after the train/test split is complete
- **Returns separate train/test sets**: Ensures no temporal overlap between training and test data

### 2. **New `split_flat_sequences()` function** (lines 164-193)
- **Consistent approach for flat models**: Also splits by session first before flattening
- **Prevents frame-level leakage**: Even for Dense models, sessions are split first

### 3. **Updated main execution** (lines 196-225)
- **Proper data loading**: Now calls the new split functions instead of windowing first
- **Clear separation**: Train and test data are properly separated before any model training

### Why This Fixes the Data Leakage

| **Before (Leaky)** | **After (Fixed)** |
|-------------------|-------------------|
| Window all data → Split | Split sessions → Window each split |
| Windows from same session in both train/test | Complete session separation |
| Temporal overlap between splits | No temporal overlap |
| Overly optimistic R² scores | Realistic performance estimates |

The fix ensures that when you evaluate your model on the test set, it's truly generalizing to new, unseen motion sequences rather than just interpolating within the same sessions.
"""
