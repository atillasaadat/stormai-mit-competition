import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from datahandler import DataHandler  # assuming datahandler.py is in your PYTHONPATH

# ---------------------------
# Helper Functions and Classes
# ---------------------------

def load_and_preprocess_data(dh, file_usage_percent, target_col, selected_feature_columns, epsilon, target_factor):
    propagated_file_ids = sorted(dh.get_all_file_ids_from_folder(dh.sat_density_omni_propagated_folder))
    logging.info(f"Total propagated files available: {len(propagated_file_ids)}")
    num_files_to_use = max(1, int(len(propagated_file_ids) * file_usage_percent))
    selected_file_ids = random.sample(propagated_file_ids, num_files_to_use)
    logging.info(f"Using {num_files_to_use} randomly selected files for training (i.e. {file_usage_percent*100:.1f}%).")
    
    dfs = []
    for file_id in selected_file_ids:
        df = dh.read_csv_data(file_id, folder=dh.sat_density_omni_propagated_folder)
        df = df[df[target_col] != 9.99e+32]
        df["file_id"] = file_id
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    data.sort_values("Timestamp", inplace=True)
    data.reset_index(drop=True, inplace=True)
    logging.info(f"Combined dataset shape: {data.shape}")
    
    # Save raw data for plotting.
    data_raw = data.copy()
    
    # Drop rows with NaNs.
    data = data.dropna()
    logging.info(f"Dataset shape after dropping NaNs: {data.shape}")
    
    # Compute log10(Altitude) for training.
    data["log_Altitude"] = np.log10(data["Altitude (km)"])
    
    logging.info("Orbit Mean Density stats: " + str(data[target_col].describe()))
    corr_df = data[[target_col] + selected_feature_columns].corr()
    logging.info(f"Correlation matrix:\n{corr_df}")
    
    # Scale input features.
    X = data[selected_feature_columns].values
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / (X_std + epsilon)
    
    # Scale target: multiply by target_factor.
    y = data[target_col].values.reshape(-1, 1) * target_factor
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    y_scaled = (y - y_mean) / (y_std + epsilon)
    
    if np.isnan(X_scaled).any() or np.isnan(y_scaled).any():
        logging.error("NaN values found in scaled data!")
        raise ValueError("Scaling produced NaN values.")
    
    file_ids = data["file_id"].values
    return X_scaled, y_scaled, X_mean, X_std, y_mean, y_std, file_ids, data_raw, target_factor

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len, file_ids=None):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.file_ids = file_ids

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        if self.file_ids is not None:
            file_id = self.file_ids[idx + self.seq_len]
            return torch.FloatTensor(x_seq), torch.FloatTensor(y_val), file_id
        else:
            return torch.FloatTensor(x_seq), torch.FloatTensor(y_val)

def get_datasets(X_scaled, y_scaled, file_ids, seq_len, split_ratio):
    split_index = int(len(X_scaled) * split_ratio)
    X_train, y_train = X_scaled[:split_index], y_scaled[:split_index]
    X_test, y_test = X_scaled[split_index:], y_scaled[split_index:]
    file_ids_train, file_ids_test = file_ids[:split_index], file_ids[split_index:]
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_len, file_ids=file_ids_test)
    return train_dataset, test_dataset, file_ids_train, file_ids_test

class PatchTST(nn.Module):
    def __init__(self, seq_len, patch_size, num_features, d_model, nhead, num_layers, dropout=0.1):
        super(PatchTST, self).__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(patch_size * num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(B, self.num_patches, self.patch_size * C)
        x = self.proj(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return out

def train_model_fn(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
    for x_batch, y_batch in progress_bar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")
    avg_loss = total_loss / len(dataloader)
    logging.info(f"Epoch {epoch+1} - Training Loss: {avg_loss:.6f}")
    return avg_loss

def evaluate_model_fn(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch, _ in tqdm(dataloader, desc="Evaluating", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            all_preds.append(output.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    mse_avg = total_loss / len(dataloader)
    rmse_avg = np.sqrt(mse_avg)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse_overall = np.mean((all_preds - all_targets) ** 2)
    rmse_overall = np.sqrt(mse_overall)
    return rmse_avg, rmse_overall

def plot_file_predictions(file_id, data_raw, model, seq_len, X_mean, X_std, y_mean, y_std,
                          target_factor, epsilon, selected_feature_columns, target_col, device):
    if device is None:
        device = next(model.parameters()).device

    file_data = data_raw[data_raw["file_id"] == file_id].copy()
    file_data.sort_values("Timestamp", inplace=True)

    num_features = len(selected_feature_columns)
    scaled_arrays = np.zeros((len(file_data), num_features))
    for col_i, col_name in enumerate(selected_feature_columns):
        if col_name == "log_Altitude":
            if "log_Altitude" not in file_data.columns:
                file_data["log_Altitude"] = np.log10(file_data["Altitude (km)"])
            raw_vals = file_data["log_Altitude"].values
        else:
            raw_vals = file_data[col_name].values
        scaled_arrays[:, col_i] = (raw_vals - X_mean[col_i]) / (X_std[col_i] + epsilon)

    predictions = []
    timestamps = []
    for i in range(len(file_data) - seq_len):
        window = scaled_arrays[i : i + seq_len]
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(window_tensor)
        pred = (pred_scaled.cpu().item() * y_std[0] + y_mean[0]) / target_factor
        predictions.append(pred)
        timestamps.append(file_data.iloc[i + seq_len]["Timestamp"])

    ground_truth = file_data[target_col].values[seq_len:]
    msis_values = file_data["MSIS Density (kg/m^3)"].values[seq_len:]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    ax1.plot(timestamps, ground_truth, label="Ground Truth Orbit Mean Density", marker="o", linestyle="-")
    ax1.plot(timestamps, predictions, label="Predicted Orbit Mean Density", marker="x", linestyle="--")
    ax1.plot(timestamps, msis_values, label="MSIS Density", marker=".", linestyle=":")
    ax1.set_ylabel("Density (kg/m^3)")
    ax1.set_title(f"File ID {file_id} - Density Predictions")
    ax1.legend()

    ax2.plot(file_data["Timestamp"], file_data["Altitude (km)"], label="Altitude (km)")
    ax2.set_ylabel("Altitude (km)")
    ax2.set_xlabel("Timestamp")
    ax2.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_random_file_id(file_ids_test, data_raw, model, seq_len, X_mean, X_std, y_mean, y_std,
                        target_factor, epsilon, selected_feature_columns, target_col, device):
    unique_test_file_ids = np.unique(file_ids_test)
    logging.info("Test set file IDs: " + ", ".join(map(str, unique_test_file_ids)))
    random_file_id = np.random.choice(unique_test_file_ids)
    logging.info(f"Plotting predictions for File ID: {random_file_id}")
    plot_file_predictions(random_file_id, data_raw, model, seq_len, X_mean, X_std, y_mean, y_std,
                          target_factor, epsilon, selected_feature_columns, target_col, device)

def run_predictions_from_csv(csv_filepath, 
                             model_path="patchtst_model.pth",
                             seq_len=10,
                             patch_size=2,
                             d_model=128,
                             nhead=4,
                             num_layers=3,
                             selected_feature_columns=["MSIS Density (kg/m^3)", "log_Altitude"],
                             device=None,
                             X_mean=None,
                             X_std=None,
                             y_mean=None,
                             y_std=None,
                             target_factor=1e12,
                             epsilon=1e-8):
    """
    Loads a CSV file and returns a prediction array using the saved PatchTST model.
    
    Parameters:
      csv_filepath (str or Path): Path to the CSV file.
      model_path (str or Path): Path to the saved model state dict.
      seq_len (int): The sliding window length (must match training).
      patch_size (int): The patch size used during training.
      d_model (int): Model embedding dimension.
      nhead (int): Number of attention heads.
      num_layers (int): Number of Transformer encoder layers.
      selected_feature_columns (list): List of features to use. For training, this should include "MSIS Density (kg/m^3)"
           and "log_Altitude". "log_Altitude" is computed from the "Altitude (km)" column.
      device (torch.device): Device to run predictions on. If None, it is determined automatically.
      X_mean, X_std, y_mean, y_std (np.array or None): Scaling parameters. If None, computed from CSV.
      target_factor (float): Factor by which target was multiplied during training.
      epsilon (float): Small constant to avoid division by zero.
      
    Returns:
      predictions (np.array): Array of predictions.
      timestamps (np.array): Array of timestamps corresponding to the predictions.
    """
    import pandas as pd
    import numpy as np
    import torch
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CSV file
    df = pd.read_csv(csv_filepath, parse_dates=["Timestamp"])
    df.sort_values("Timestamp", inplace=True)
    
    # Compute log10(Altitude) if not present
    if "log_Altitude" not in df.columns:
        df["log_Altitude"] = np.log10(df["Altitude (km)"])
    
    # Build feature matrix using selected columns.
    X_input = df[selected_feature_columns].values
    
    # If scaling parameters are not provided, compute them from the file.
    if X_mean is None:
        X_mean = np.mean(X_input, axis=0)
    if X_std is None:
        X_std = np.std(X_input, axis=0)
    X_scaled = (X_input - X_mean) / (X_std + epsilon)
    
    # For this function, we assume no target is needed from the CSV (we're only predicting).
    num_features = len(selected_feature_columns)
    
    # Create model instance with provided hyperparameters.
    from torch import nn
    class PatchTST(nn.Module):
        def __init__(self, seq_len, patch_size, num_features, d_model, nhead, num_layers, dropout=0.1):
            super(PatchTST, self).__init__()
            self.patch_size = patch_size
            self.num_patches = seq_len // patch_size
            self.proj = nn.Linear(patch_size * num_features, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                       dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            B, L, C = x.shape
            x = x.reshape(B, self.num_patches, self.patch_size * C)
            x = self.proj(x)
            x = self.transformer_encoder(x)
            x = x[:, -1, :]
            out = self.fc(x)
            return out
    
    model_instance = PatchTST(seq_len=seq_len, patch_size=patch_size, num_features=num_features,
                              d_model=d_model, nhead=nhead, num_layers=num_layers)
    model_instance.to(device)
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()
    
    predictions = []
    timestamps = []
    num_windows = len(X_scaled) - seq_len
    # If scaling parameters for target are not provided, compute from the current CSV's target (if available).
    # Here we assume that for prediction, you use provided values.
    if y_mean is None or y_std is None:
        # Optionally, you might load training values instead.
        # For now, we set defaults.
        y_mean = np.array([0.0])
        y_std = np.array([1.0])
    
    for i in range(num_windows):
        window = X_scaled[i:i+seq_len]  # shape (seq_len, num_features)
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)  # (1, seq_len, num_features)
        with torch.no_grad():
            pred_scaled = model_instance(window_tensor)
        # Inverse scaling: the model output is in scaled space.
        pred_value = (pred_scaled.cpu().item() * y_std[0] + y_mean[0]) / target_factor
        predictions.append(pred_value)
        timestamps.append(df.iloc[i+seq_len]["Timestamp"])
    
    return np.array(predictions), np.array(timestamps)

# ---------------------------
# Main Function
# ---------------------------
if __name__ == "__main__":
    # Configuration variables declared in main
    DATA_PATHS = {
        "omni2_folder": Path("./data/omni2"),
        "initial_state_folder": Path("./data/initial_state"),
        "sat_density_folder": Path("./data/sat_density"),
        "forcasted_omni2_folder": Path("./data/forcasted_omni2"),
        "sat_density_omni_forcasted_folder": Path("./data/sat_density_omni_forcasted"),
        "sat_density_omni_propagated_folder": Path("./data/sat_density_omni_propagated"),
    }
    FILE_USAGE_PERCENT = 0.5
    TARGET_COL = "Orbit Mean Density (kg/m^3)"
    SELECTED_FEATURE_COLUMNS = ["MSIS Density (kg/m^3)", "Latitude (deg)", "Longitude (deg)", "log_Altitude"]
    SEQ_LEN = 10
    SPLIT_RATIO = 0.8
    TARGET_FACTOR = 1e-12
    EPSILON = 1e-8

    # Setup logger (if desired, you can configure logging here as well)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Initialize DataHandler
    dh_local = DataHandler(logger, **DATA_PATHS)

    # Load and preprocess data
    X_scaled, y_scaled, X_mean, X_std, y_mean, y_std, file_ids, data_raw, target_factor = load_and_preprocess_data(
        dh_local, FILE_USAGE_PERCENT, TARGET_COL, SELECTED_FEATURE_COLUMNS, EPSILON, TARGET_FACTOR
    )

    # Build datasets and dataloaders
    train_dataset, test_dataset, file_ids_train, file_ids_test = get_datasets(X_scaled, y_scaled, file_ids, SEQ_LEN, SPLIT_RATIO)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Setup device and model parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    num_features = X_scaled.shape[1]
    patch_size = 2
    d_model = 128
    nhead = 4
    num_layers = 3

    model = PatchTST(seq_len=SEQ_LEN, patch_size=patch_size, num_features=num_features,
                     d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 15
    for epoch in range(num_epochs):
        train_model_fn(model, train_loader, optimizer, criterion, device, epoch)
        rmse_avg, rmse_overall = evaluate_model_fn(model, test_loader, criterion, device)
        logger.info(f"Epoch {epoch+1} - Test RMSE (batch avg): {rmse_avg:.6f}, Test RMSE (overall): {rmse_overall:.6f}")

    rmse_avg, rmse_overall = evaluate_model_fn(model, test_loader, criterion, device)
    logger.info(f"Final Test RMSE (batch avg): {rmse_avg:.6f}")
    logger.info(f"Final Test RMSE (overall): {rmse_overall:.6f}")

    # Save the trained model
    model_save_path = Path("density_prediction_patchtst_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    # Optionally, plot predictions for a random test file.
    plot_random_file_id(file_ids_test, data_raw, model, SEQ_LEN, X_mean, X_std, y_mean, y_std, TARGET_FACTOR, EPSILON, SELECTED_FEATURE_COLUMNS, TARGET_COL, device)
    from IPython import embed; embed(); quit()