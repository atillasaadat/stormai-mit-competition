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
import json

from datahandler import DataHandler  # assuming datahandler.py is in your PYTHONPATH

# ---------------------------
# Helper Functions and Classes
# ---------------------------
def load_and_preprocess_data(dh, file_usage_percent, target_col, selected_feature_columns, epsilon):
    # Get list of files in the propagated folder.
    propagated_file_ids = sorted(dh.get_all_file_ids_from_folder(dh.sat_density_omni_propagated_folder))
    logging.info(f"Total propagated files available: {len(propagated_file_ids)}")
    num_files_to_use = max(1, int(len(propagated_file_ids) * file_usage_percent))
    selected_file_ids = np.random.choice(propagated_file_ids, size=num_files_to_use, replace=False).tolist()
    logging.info(f"Using {num_files_to_use} randomly selected files for training ({file_usage_percent*100:.1f}%).")
    
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
    
    # Compute log_Altitude so both data and data_raw have it.
    data["log_Altitude"] = np.log10(data["Altitude (km)"])
    data_raw = data.copy()
    
    data = data.dropna()
    logging.info(f"Dataset shape after dropping NaNs: {data.shape}")
    
    logging.info("Orbit Mean Density stats: " + str(data[target_col].describe()))
    corr_df = data[[target_col] + selected_feature_columns].corr()
    logging.info(f"Correlation matrix:\n{corr_df}")
    
    # For inputs, use the raw selected features (do not transform the target).
    X_raw = data[selected_feature_columns].values
    # For the target, compute the correction ratio: truth density / MSIS density.
    msis_density = data["MSIS Density (kg/m^3)"].values.reshape(-1, 1)
    y_ratio = data[target_col].values.reshape(-1, 1) / msis_density

    file_ids = data["file_id"].values
    return X_raw, y_ratio, file_ids, data_raw

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

def get_datasets(X_raw, y_ratio, file_ids, seq_len, split_ratio, epsilon):
    # First split into train and test; then compute normalization parameters on train set only.
    split_index = int(len(X_raw) * split_ratio)
    X_train_raw, y_train = X_raw[:split_index], y_ratio[:split_index]
    X_test_raw, y_test = X_raw[split_index:], y_ratio[split_index:]
    file_ids_train, file_ids_test = file_ids[:split_index], file_ids[split_index:]
    
    # Compute normalization parameters on training data only.
    X_mean = np.mean(X_train_raw, axis=0)
    X_std = np.std(X_train_raw, axis=0)
    # Apply same transformation to both training and test data.
    X_train = (X_train_raw - X_mean) / (X_std + epsilon)
    X_test = (X_test_raw - X_mean) / (X_std + epsilon)
    
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_len, file_ids=file_ids_test)
    return train_dataset, test_dataset, file_ids_train, file_ids_test, X_mean, X_std

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
        # Reshape into patches
        x = x.reshape(B, self.num_patches, self.patch_size * C)
        x = self.proj(x)
        x = self.transformer_encoder(x)
        # Use the representation from the last patch.
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

def plot_file_predictions(file_id, data_raw, model, seq_len, selected_feature_columns, target_col, device, X_mean, X_std, epsilon):
    file_data = data_raw[data_raw["file_id"] == file_id].copy()
    file_data.sort_values("Timestamp", inplace=True)
    if "log_Altitude" not in file_data.columns:
        file_data["log_Altitude"] = np.log10(file_data["Altitude (km)"])
    
    raw_arrays = file_data[selected_feature_columns].values
    # Apply normalization using training-set mean and std.
    X_scaled = (raw_arrays - X_mean) / (X_std + epsilon)
    
    predictions = []
    timestamps = []
    for i in range(len(file_data) - seq_len):
        window = X_scaled[i : i + seq_len]
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            pred_ratio = model(window_tensor).cpu().item()
        msis_val = file_data.iloc[i + seq_len]["MSIS Density (kg/m^3)"]
        pred_density = pred_ratio * msis_val
        predictions.append(pred_density)
        timestamps.append(file_data.iloc[i + seq_len]["Timestamp"])
    
    ground_truth = file_data[target_col].values[seq_len:]
    msis_values = file_data["MSIS Density (kg/m^3)"].values[seq_len:]
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    ax1.plot(timestamps, ground_truth, label="Ground Truth", marker="o", linestyle="-")
    ax1.plot(timestamps, predictions, label="Predicted Corrected Density", marker="x", linestyle="--")
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

def plot_random_file_id(file_ids_test, data_raw, model, seq_len, selected_feature_columns, target_col, device, X_mean, X_std, epsilon):
    unique_test_file_ids = np.unique(file_ids_test)
    logging.info("Test set file IDs: " + ", ".join(map(str, unique_test_file_ids)))
    random_file_id = np.random.choice(unique_test_file_ids)
    logging.info(f"Plotting predictions for File ID: {random_file_id}")
    plot_file_predictions(random_file_id, data_raw, model, seq_len, selected_feature_columns, target_col, device, X_mean, X_std, epsilon)

# ---------------------------
# Main Training Pipeline
# ---------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    DATA_PATHS = {
        "omni2_folder": Path("./data/omni2"),
        "initial_state_folder": Path("./data/initial_state"),
        "sat_density_folder": Path("./data/sat_density"),
        "forcasted_omni2_folder": Path("./data/forcasted_omni2"),
        "sat_density_omni_forcasted_folder": Path("./data/sat_density_omni_forcasted"),
        "sat_density_omni_propagated_folder": Path("./data/sat_density_omni_propagated"),
    }
    FILE_USAGE_PERCENT = 0.8
    TARGET_COL = "Orbit Mean Density (kg/m^3)"
    SELECTED_FEATURE_COLUMNS = ["MSIS Density (kg/m^3)", "Latitude (deg)", "Longitude (deg)", "log_Altitude"]
    SEQ_LEN = 50
    SPLIT_RATIO = 0.8
    EPSILON = 1e-8

    # Initialize DataHandler
    dh = DataHandler(logger, **DATA_PATHS)
    X_raw, y_ratio, file_ids, data_raw = load_and_preprocess_data(
        dh, FILE_USAGE_PERCENT, TARGET_COL, SELECTED_FEATURE_COLUMNS, EPSILON
    )

    train_dataset, test_dataset, file_ids_train, file_ids_test, X_mean, X_std = get_datasets(
        X_raw, y_ratio, file_ids, SEQ_LEN, SPLIT_RATIO, EPSILON
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    num_features = X_raw.shape[1]
    patch_size = 5
    d_model = 128
    nhead = 4
    num_layers = 3

    model = PatchTST(seq_len=SEQ_LEN, patch_size=patch_size, num_features=num_features,
                     d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.to(device)

    # Use AdamW with weight decay and switch to Huber loss for robustness.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()

    # Learning rate scheduler that reduces LR when validation loss plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model_fn(model, train_loader, optimizer, criterion, device, epoch)
        rmse_avg, rmse_overall = evaluate_model_fn(model, test_loader, criterion, device)
        logging.info(f"Epoch {epoch+1} - Test RMSE (batch avg): {rmse_avg:.6f}, Test RMSE (overall): {rmse_overall:.6f}")
        scheduler.step(train_loss)

    rmse_avg, rmse_overall = evaluate_model_fn(model, test_loader, criterion, device)
    logging.info(f"Final Test RMSE (batch avg): {rmse_avg:.6f}")
    logging.info(f"Final Test RMSE (overall): {rmse_overall:.6f}")

    model_save_path = Path("density_prediction_patchtst_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    plot_random_file_id(file_ids_test, data_raw, model, SEQ_LEN, SELECTED_FEATURE_COLUMNS, TARGET_COL, device, X_mean, X_std, EPSILON)
    from IPython import embed; embed(); quit()