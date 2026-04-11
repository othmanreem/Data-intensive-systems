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

REPO_ROOT    = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_DIR     = os.path.join(REPO_ROOT, 'Datasets_all')
KINECT_DATA_PATH = os.path.join(DATA_DIR, 'kinect_good_preprocessed')

# sequences contain list of tuples (X,y)
sequences, file_names = load_all_sequences(KINECT_DATA_PATH)

# Frame-level flat data  (for Dense models)
X_flat, y_flat = flatten_sequences(sequences)
print(f"\nFlat dataset:      X={X_flat.shape}  y={y_flat.shape}")

# Windowed sequences  (for Conv1D / LSTM / GRU models)
WINDOW_SIZE = 30
X_seq, y_seq = make_windowed_sequences(sequences, window_size=WINDOW_SIZE, stride=5)
y_seq_last = y_seq[:, -1, :]                         # (N, 13)
print(f"Windowed dataset:  X={X_seq.shape}  y_last={y_seq_last.shape}")

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
