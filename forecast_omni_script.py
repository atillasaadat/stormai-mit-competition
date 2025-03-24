#!/usr/bin/env python
"""
Forecasting Space Weather using a Transformer-based Model.
This script trains a PatchTST model to forecast 'ap_index_nT' and 'f10.7_index'
from space weather data, and then validates forecasts against historical data.
"""

import os
# Workaround for OpenMP error (set before any other imports)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

# ============================
# Logging configuration
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================
# Model and Dataset Definitions
# ============================
class PatchTST(nn.Module):
    """
    A simplified PatchTST model.
    """
    def __init__(self, input_size, patch_size, d_model, n_heads, n_layers,
                 forecast_horizon, target_size, dropout=0.1):
        super(PatchTST, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.target_size = target_size

        logger.debug("Initializing PatchTST model.")
        self.proj = nn.Conv1d(in_channels=input_size, out_channels=d_model,
                              kernel_size=patch_size, stride=patch_size)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.fc = nn.Linear(d_model, forecast_horizon * target_size)

    def forward(self, x):
        batch_size = x.size(0)
        # Switch dimensions for Conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.transformer_encoder(x)
        x_last = x[:, -1, :]
        out = self.fc(x_last)
        out = out.view(batch_size, self.forecast_horizon, self.target_size)
        return out

class TimeSeriesDataset(data.Dataset):
    """
    Creates sliding-window samples from the time series data.
    """
    def __init__(self, X, Y, input_window, forecast_horizon):
        self.X = X
        self.Y = Y
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.length = len(X) - input_window - forecast_horizon + 1
        logger.debug(f"TimeSeriesDataset initialized with {self.length} samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.input_window]
        y = self.Y[idx + self.input_window : idx + self.input_window + self.forecast_horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ============================
# Data Handling
# ============================
class DataHandler:
    def __init__(self, omni_folder, initial_state_folder, sat_density_folder, forecasted_omni2_folder):
        self.omni_folder = omni_folder
        self.initial_state_folder = initial_state_folder
        self.sat_density_folder = sat_density_folder
        self.forecasted_omni2_folder = forecasted_omni2_folder
        self._read_initial_states()

    def _read_initial_states(self):
        dfs = []
        for file in self.initial_state_folder.iterdir():
            if file.suffix == '.csv':
                dfs.append(pd.read_csv(file))
        self.initial_states = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded initial state data with {len(self.initial_states)} rows.")

    def read_omni_data(self, file_id):
        file_id_str = f"{file_id:05d}"
        for file in self.omni_folder.iterdir():
            if file.suffix == '.csv' and file_id_str in file.stem:
                logger.debug(f"Reading OMNI file for File ID {file_id}.")
                return pd.read_csv(file)
        raise FileNotFoundError(f"File with ID {file_id} not found in {self.omni_folder}")

    def read_sat_density_data(self, file_id):
        file_id_str = f"{file_id:05d}"
        for file in self.sat_density_folder.iterdir():
            if file.suffix == '.csv' and file_id_str in file.stem:
                logger.debug(f"Reading Sat density file for File ID {file_id}.")
                return pd.read_csv(file)
        raise FileNotFoundError(f"File with ID {file_id} not found in {self.sat_density_folder}")

# ============================
# Training and Evaluation Functions
# ============================
def train_model(model, train_loader, val_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device)
    logger.info("Starting training loop.")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")):
            if torch.isnan(x).any() or torch.isnan(y).any():
                logger.error(f"NaNs detected in batch {batch_idx} of epoch {epoch+1}")
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            if torch.isnan(loss):
                logger.error(f"Loss is NaN at epoch {epoch+1}, batch {batch_idx}. Aborting training.")
                return
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")

        model.eval()
        val_loss = 0.0
        for batch_idx, (x, y) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation")):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.6f}")

def evaluate_model(model, test_loader, device, collect_outputs=False):
    model.eval()
    total_mse, total_mae = 0.0, 0.0
    preds_list, trues_list = [], []

    with torch.no_grad():
        for data_batch, target_batch in test_loader:
            data_batch, target_batch = data_batch.to(device), target_batch.to(device)
            output = model(data_batch)
            mse = F.mse_loss(output, target_batch, reduction='sum').item()
            mae = F.l1_loss(output, target_batch, reduction='sum').item()
            total_mse += mse
            total_mae += mae
            if collect_outputs:
                preds_list.append(output.cpu())
                trues_list.append(target_batch.cpu())

    total_samples = len(test_loader.dataset)
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    if collect_outputs:
        preds = torch.cat(preds_list, dim=0)
        trues = torch.cat(trues_list, dim=0)
    else:
        preds, trues = None, None
    return avg_mse, avg_mae, preds, trues

def forecast(model, input_sequence, device):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)
    return output.squeeze(0).cpu().numpy()

# ============================
# Plotting Functions
# ============================
def plot_combined_forecast(history_timestamps, history_values, forecast_timestamps, forecast_values,
                           ylabel, title, true_forecast=None):
    logger.debug("Plotting combined forecast.")
    history_timestamps = pd.to_datetime(history_timestamps)
    forecast_timestamps = pd.to_datetime(forecast_timestamps)

    fig, ax = plt.subplots(figsize=(12, 6))
    if true_forecast is not None:
        ax.plot(forecast_timestamps, true_forecast, label="True Forecast",
                marker='o', linestyle='--', color='green', zorder=1, alpha=0.5)
    ax.plot(history_timestamps, history_values, label="Historical",
            marker='o', color='blue', zorder=2)
    ax.plot(forecast_timestamps, forecast_values, label="Forecast",
            marker='x', color='red', zorder=2)

    ax.set_xlabel("Timestamp")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_forecast_for_file(omni_df, predicted_sat_df, target, history_days=60):
    """
    Plots the historical values for a target from omni_df (for the given number of days preceding the
    forecast) and the predicted values from predicted_sat_df (at the sat density timestamps).
    """
    forecast_start = predicted_sat_df["Timestamp"].min()
    history_start = forecast_start - pd.Timedelta(days=history_days)
    history_data = omni_df[(omni_df["Timestamp"] >= history_start) & (omni_df["Timestamp"] < forecast_start)]

    plt.figure(figsize=(12, 6))
    plt.plot(history_data["Timestamp"], history_data[target], label="Historical " + target,
             marker='o', color="blue")
    plt.plot(predicted_sat_df["Timestamp"], predicted_sat_df[target + "_pred"],
             label="Predicted " + target, marker='x', linestyle="--", color="red")
    plt.xlabel("Timestamp")
    plt.ylabel(target)
    plt.title(f"Historical and Predicted {target} for File ID {predicted_sat_df['File_ID'].iloc[0] if 'File_ID' in predicted_sat_df.columns else 'N/A'}")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def predict_for_each_timestamp_in_sat_file(
    model,
    data_handler,
    file_id,
    omni_df,
    input_features,
    device,
    input_window=100
):
    """
    For a given file_id, read the satellite density timestamps.
    For each timestamp T in that file:
      1) Gather the preceding 'input_window' rows from omni_df (strictly before or up to T)
      2) Run a single-step forecast to get ap_index_nT and f10.7_index at time T
      3) Store the forecast in new columns (ap_index_nT_pred, f10.7_index_pred)
    """
    sat_df = data_handler.read_sat_density_data(file_id).copy()
    sat_df["Timestamp"] = pd.to_datetime(sat_df["Timestamp"])
    sat_df.sort_values("Timestamp", inplace=True)
    sat_df.reset_index(drop=True, inplace=True)
    sat_df["File_ID"] = file_id  # add file_id for plotting

    ap_preds = []
    f107_preds = []

    for i, row in sat_df.iterrows():
        t = row["Timestamp"]
        mask = (omni_df["Timestamp"] <= t)
        sub_omni = omni_df[mask].tail(input_window)

        if len(sub_omni) < input_window:
            ap_preds.append(np.nan)
            f107_preds.append(np.nan)
            continue

        input_seq = sub_omni[input_features].values
        with torch.no_grad():
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            out = model(input_tensor)
            out = out.detach().cpu().numpy().squeeze(0)
        if len(out.shape) == 2 and out.shape[0] > 1:
            out = out[0]
        ap_preds.append(out[0])
        f107_preds.append(out[1])

    sat_df["ap_index_nT_pred"] = ap_preds
    sat_df["f10.7_index_pred"] = f107_preds
    return sat_df

def plot_results_for_random_file(model, data_handler, omni_df, input_features, device, input_window, file_ids):
    """
    Randomly selects a file_id from file_ids, forecasts each timestamp in the corresponding
    satellite density file, and produces plots using all available plotting functions.
    """
    random_file_id = np.random.choice(file_ids)
    logger.info(f"Randomly selected File ID: {random_file_id}")

    # Get predictions for every timestamp in the satellite file.
    predicted_sat_df = predict_for_each_timestamp_in_sat_file(
        model, data_handler, random_file_id, omni_df, input_features, device, input_window
    )

    # Plot using the dedicated forecast plot for each target.
    plot_forecast_for_file(omni_df, predicted_sat_df, target="ap_index_nT", history_days=60)
    plot_forecast_for_file(omni_df, predicted_sat_df, target="f10.7_index", history_days=60)

    # Also plot a combined forecast (historical vs. forecast) for each target.
    # For ap_index_nT:
    forecast_timestamps = pd.to_datetime(predicted_sat_df["Timestamp"])
    forecast_values = predicted_sat_df["ap_index_nT_pred"]
    forecast_start = forecast_timestamps.min()
    history_start = forecast_start - pd.Timedelta(days=60)
    history_data = omni_df[(omni_df["Timestamp"] >= history_start) & (omni_df["Timestamp"] < forecast_start)]
    history_timestamps = pd.to_datetime(history_data["Timestamp"])
    history_values = history_data["ap_index_nT"]
    plot_combined_forecast(history_timestamps, history_values, forecast_timestamps, forecast_values,
                           ylabel="ap_index_nT",
                           title=f"Combined Forecast for ap_index_nT (File ID {random_file_id})",
                           true_forecast=None)

    # For f10.7_index:
    forecast_values_f107 = predicted_sat_df["f10.7_index_pred"]
    history_values_f107 = history_data["f10.7_index"]
    plot_combined_forecast(history_timestamps, history_values_f107, forecast_timestamps, forecast_values_f107,
                           ylabel="f10.7_index",
                           title=f"Combined Forecast for f10.7_index (File ID {random_file_id})",
                           true_forecast=None)

# ============================
# Main Execution Function
# ============================
def main():
    # Parameters
    batch_size = 32
    patch_size = 10
    d_model = 64
    n_heads = 8
    n_layers = 6
    dropout = 0.1
    epochs = 10
    lr = 1e-4
    input_window = 100
    load_percentage = 0.1

    # Setup paths (update as needed)
    if "google.colab" in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        storm_ai_path = Path("/content/drive/My Drive/STORM AI")
    else:
        storm_ai_path = Path("./")
    omni_folder = storm_ai_path / "data/omni2"
    initial_state_folder = storm_ai_path / "data/initial_state"
    sat_density_folder = storm_ai_path / "data/sat_density"
    forecasted_omni2_folder = storm_ai_path / "data/forcasted_omni2"

    # Data handling
    data_handler = DataHandler(omni_folder, initial_state_folder, sat_density_folder, forecasted_omni2_folder)
    logger.info(f"Initial state columns: {data_handler.initial_states.columns.tolist()}")

    file_ids = data_handler.initial_states["File ID"].unique()
    logger.info(f"Total file IDs available: {len(file_ids)}")
    if load_percentage < 1.0:
        file_ids = np.random.choice(file_ids, size=int(load_percentage * len(file_ids)), replace=False)
        logger.info(f"Randomly selected {len(file_ids)} file IDs ({load_percentage*100:.0f}%) for loading.")

    omni_dfs, sat_density_dfs = [], []
    total_ids = len(file_ids)
    next_threshold = 0.05

    for i, fid in enumerate(file_ids):
        try:
            omni_dfs.append(data_handler.read_omni_data(fid))
        except FileNotFoundError:
            logger.warning(f"OMNI file for File ID {fid} not found; skipping.")
        try:
            sat_density_dfs.append(data_handler.read_sat_density_data(fid))
        except FileNotFoundError:
            logger.warning(f"Sat density file for File ID {fid} not found; skipping.")
        if (i + 1) / total_ids >= next_threshold:
            logger.info(f"{int(next_threshold * 100)}% complete")
            next_threshold += 0.05

    if len(omni_dfs) == 0 or len(sat_density_dfs) == 0:
        raise ValueError("No valid OMNI or sat_density files were loaded.")

    omni_df = pd.concat(omni_dfs, ignore_index=True)
    sat_density_df = pd.concat(sat_density_dfs, ignore_index=True)
    logger.info(f"Loaded {len(omni_df)} rows of OMNI data and {len(sat_density_df)} rows of sat density data.")

    # Preprocessing
    omni_df['Timestamp'] = pd.to_datetime(omni_df['Timestamp'])
    omni_df.sort_values('Timestamp', inplace=True)
    sat_density_df['Timestamp'] = pd.to_datetime(sat_density_df['Timestamp'])
    sat_density_df.sort_values('Timestamp', inplace=True)

    # Feature selection and cleaning
    ap_features = [
        "Kp_index", "Dst_index_nT", "AU_index_nT", "AL_index_nT", "AE_index_nT",
        "SW_Plasma_Speed_km_s", "SW_Proton_Density_N_cm3", "SW_Plasma_Temperature_K",
        "Scalar_B_nT", "Vector_B_Magnitude_nT", "BX_nT_GSE_GSM", "BY_nT_GSE", "BZ_nT_GSE",
        "Plasma_Beta", "Flow_pressure"
    ]
    f107_features = ["Bartels_rotation_number", "R_Sunspot_No", "Lyman_alpha"]
    candidate_features = list(set(ap_features).union(set(f107_features)))
    candidate_features = [f for f in candidate_features if f in omni_df.columns]

    cleaned_features = []
    for f in candidate_features:
        if omni_df[f].isna().all() or omni_df[f].nunique() <= 1:
            logger.debug(f"Excluding feature {f}.")
            continue
        cleaned_features.append(f)
    input_features = cleaned_features
    target_features = ["ap_index_nT", "f10.7_index"]

    logger.info(f"Selected input features: {input_features}")
    logger.info(f"Target features: {target_features}")

    # Fill NaNs and normalize
    for col in input_features:
        if omni_df[col].isna().any():
            median_val = omni_df[col].median()
            omni_df[col] = omni_df[col].fillna(median_val)
    input_data = omni_df[input_features]
    input_data = (input_data - input_data.mean()) / input_data.std()
    omni_df[input_features] = input_data

    # Create datasets
    X = omni_df[input_features].values
    Y = omni_df[target_features].values
    logger.info(f"Input shape: {X.shape}, Target shape: {Y.shape}")

    unique_timestamps = sat_density_df["Timestamp"].drop_duplicates().reset_index(drop=True)
    forecast_horizon = len(unique_timestamps)
    logger.info(f"Using input_window={input_window} and forecast_horizon={forecast_horizon}")

    split_idx = int(0.8 * len(X))
    train_X = X[:split_idx]
    train_Y = Y[:split_idx]
    test_X = X[split_idx - input_window - forecast_horizon + 1:]
    test_Y = Y[split_idx - input_window - forecast_horizon + 1:]

    train_dataset = TimeSeriesDataset(train_X, train_Y, input_window, forecast_horizon)
    test_dataset = TimeSeriesDataset(test_X, test_Y, input_window, forecast_horizon)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")

    # Instantiate and train model
    model = PatchTST(input_size=len(input_features), patch_size=patch_size,
                     d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                     forecast_horizon=forecast_horizon, target_size=len(target_features),
                     dropout=dropout)
    logger.info("Model instantiated.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device.")
    else:
        raise ValueError("CUDA not available")
    
    logger.info("Starting training...")
    train_model(model, train_loader, test_loader, epochs, lr, device)

    test_mse, test_mae, _, _ = evaluate_model(model, test_loader, device, collect_outputs=False)
    logger.info(f"Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")

    # Example forecast plotting from a sample in the test dataset.
    sample_index = 0  # choose an index as needed
    test_offset = split_idx - input_window - forecast_horizon + 1
    global_index = test_offset + sample_index

    history_timestamps = pd.to_datetime(omni_df['Timestamp'].iloc[global_index : global_index + input_window])
    history_ap = omni_df['ap_index_nT'].iloc[global_index : global_index + input_window]
    history_f107 = omni_df['f10.7_index'].iloc[global_index : global_index + input_window]

    forecast_timestamps = pd.to_datetime(omni_df['Timestamp'].iloc[global_index + input_window : global_index + input_window + forecast_horizon])
    true_forecast_ap = omni_df['ap_index_nT'].iloc[global_index + input_window : global_index + input_window + forecast_horizon]
    true_forecast_f107 = omni_df['f10.7_index'].iloc[global_index + input_window : global_index + input_window + forecast_horizon]

    sample_x, _ = test_dataset[sample_index]
    sample_x = sample_x.unsqueeze(0).to(device)
    pred_sample = model(sample_x).cpu().detach().numpy().squeeze(0)
    pred_ap = pred_sample[:, 0]
    pred_f107 = pred_sample[:, 1]

    plot_combined_forecast(history_timestamps, history_ap, forecast_timestamps, pred_ap,
                           ylabel="ap_index_nT", title="Historical and Forecasted ap_index_nT",
                           true_forecast=true_forecast_ap)
    plot_combined_forecast(history_timestamps, history_f107, forecast_timestamps, pred_f107,
                           ylabel="f10.7_index", title="Historical and Forecasted f10.7_index",
                           true_forecast=true_forecast_f107)

    # --- NEW: Plot results for a randomly selected test file_id ---
    plot_results_for_random_file(model, data_handler, omni_df, input_features, device, input_window, file_ids)
    from IPython import embed; embed(); quit()

    # Save the model.
    model_save_path = Path("./submission/forcast_omni_patchtst_model.pth")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
