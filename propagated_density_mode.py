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

from datahandler import DataHandler  # assuming datahandler.py is in your PYTHONPATH

# ---------------------------
# 1. Setup Data Paths, Logger, and User Options
# ---------------------------
data_paths = {
    "omni2_folder": Path("./data/omni2"),
    "initial_state_folder": Path("./data/initial_state"),
    "sat_density_folder": Path("./data/sat_density"),
    "forcasted_omni2_folder": Path("./data/forcasted_omni2"),
    "sat_density_omni_forcasted_folder": Path("./data/sat_density_omni_forcasted"),
    "sat_density_omni_propagated_folder": Path("./data/sat_density_omni_propagated"),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
dh = DataHandler(logger, **data_paths)

# User options:
file_usage_percent = 0.05  # Use only 5% of the available files
target_col = "Orbit Mean Density (kg/m^3)"  # Column to predict

# Include more features: MSIS, Latitude, Longitude, Altitude.
selected_feature_columns = [
    "MSIS Density (kg/m^3)",
    "Latitude (deg)",
    "Longitude (deg)",
    "Altitude (km)"
]

# ---------------------------
# 2. Load and Preprocess Data
# ---------------------------
propagated_file_ids = sorted(dh.get_all_file_ids_from_folder(dh.sat_density_omni_propagated_folder))
logger.info(f"Total propagated files available: {len(propagated_file_ids)}")

num_files_to_use = max(1, int(len(propagated_file_ids) * file_usage_percent))
selected_file_ids = propagated_file_ids[:num_files_to_use]
logger.info(f"Using {num_files_to_use} files for training (i.e. {file_usage_percent*100:.1f}%).")

dfs = []
for file_id in selected_file_ids:
    df = dh.read_csv_data(file_id, folder=dh.sat_density_omni_propagated_folder)
    df = df[df[target_col] != 9.99e+32]
    df["file_id"] = file_id
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data.sort_values("Timestamp", inplace=True)
data.reset_index(drop=True, inplace=True)
logger.info(f"Combined dataset shape: {data.shape}")

# Save raw data for plotting and inspection.
data_raw = data.copy()

# Drop rows with NaNs in selected features or target.
data = data.dropna(subset=selected_feature_columns + [target_col])
logger.info(f"Dataset shape after dropping NaNs: {data.shape}")

# ---------------------------
# 2b. Optional: Print stats to check variability.
# ---------------------------
logger.info("Orbit Mean Density stats: " + str(data[target_col].describe()))
corr_df = data[[target_col] + selected_feature_columns].corr()
logger.info(f"Correlation matrix:\n{corr_df}")

# ---------------------------
# 3. Manual Scaling
# ---------------------------
epsilon = 1e-8
X = data[selected_feature_columns].values
y = data[target_col].values.reshape(-1, 1)

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / (X_std + epsilon)

y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y_scaled = (y - y_mean) / (y_std + epsilon)

if np.isnan(X_scaled).any() or np.isnan(y_scaled).any():
    logger.error("NaN values found in scaled data!")
    raise ValueError("Scaling produced NaN values.")

file_ids = data["file_id"].values

# ---------------------------
# 4. Create a Time Series Dataset
# ---------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len, file_ids=None):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.file_ids = file_ids
    
    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        if self.file_ids is not None:
            file_id = self.file_ids[idx + self.seq_len]
            return torch.FloatTensor(x_seq), torch.FloatTensor(y_val), file_id
        else:
            return torch.FloatTensor(x_seq), torch.FloatTensor(y_val)

seq_len = 10
split_ratio = 0.8
split_index = int(len(X_scaled) * split_ratio)
X_train, y_train = X_scaled[:split_index], y_scaled[:split_index]
X_test, y_test = X_scaled[split_index:], y_scaled[split_index:]
file_ids_train, file_ids_test = file_ids[:split_index], file_ids[split_index:]

train_dataset = TimeSeriesDataset(X_train, y_train, seq_len)
test_dataset = TimeSeriesDataset(X_test, y_test, seq_len, file_ids=file_ids_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------------
# 5. Define the PatchTST-like Model
# ---------------------------
class PatchTST(nn.Module):
    def __init__(self, seq_len, patch_size, num_features, d_model, nhead, num_layers, dropout=0.1):
        super(PatchTST, self).__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(patch_size * num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
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

# ---------------------------
# 6. Training Setup and Loop with tqdm Logging
# ---------------------------
num_features = X_train.shape[1]
patch_size = 2
d_model = 128
nhead = 4
num_layers = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = PatchTST(seq_len=seq_len, patch_size=patch_size, num_features=num_features,
                 d_model=d_model, nhead=nhead, num_layers=num_layers)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

def train_model(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
    for batch_idx, (x_batch, y_batch) in enumerate(progress_bar):
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
    logger.info(f"Epoch {epoch+1} - Training Loss: {avg_loss:.6f}")
    return avg_loss

def evaluate_model(model, dataloader, criterion, device):
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

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion, device, epoch)
    rmse_avg, rmse_overall = evaluate_model(model, test_loader, criterion, device)
    logger.info(f"Epoch {epoch+1} - Test RMSE (batch avg): {rmse_avg:.6f}, Test RMSE (overall): {rmse_overall:.6f}")

rmse_avg, rmse_overall = evaluate_model(model, test_loader, criterion, device)
logger.info(f"Final Test RMSE (batch avg): {rmse_avg:.6f}")
logger.info(f"Final Test RMSE (overall): {rmse_overall:.6f}")

# ---------------------------
# 7. Plotting Function with LLA Subplot
# ---------------------------
def plot_file_predictions(
    file_id,
    data_raw,
    model,
    seq_len,
    X_mean,
    X_std,
    y_mean,
    y_std,
    epsilon=1e-8,
    selected_feature_columns=None,
    target_col="Orbit Mean Density (kg/m^3)",
    device=None,
):
    # If no device is provided, get it from the model.
    if device is None:
        device = next(model.parameters()).device

    if selected_feature_columns is None:
        selected_feature_columns = [
            "MSIS Density (kg/m^3)",
            "Latitude (deg)",
            "Longitude (deg)",
            "Altitude (km)"
        ]

    file_data = data_raw[data_raw["file_id"] == file_id].copy()
    file_data.sort_values("Timestamp", inplace=True)

    num_features = len(selected_feature_columns)
    scaled_arrays = np.zeros((len(file_data), num_features))
    for col_i, col_name in enumerate(selected_feature_columns):
        raw_vals = file_data[col_name].values
        scaled_arrays[:, col_i] = (raw_vals - X_mean[col_i]) / (X_std[col_i] + epsilon)

    predictions = []
    timestamps = []
    for i in range(len(file_data) - seq_len):
        window = scaled_arrays[i : i + seq_len]  # (seq_len, num_features)
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)  # (1, seq_len, num_features)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(window_tensor)
        pred = pred_scaled.cpu().item() * y_std[0] + y_mean[0]
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

    ax2.plot(file_data["Timestamp"], file_data["Latitude (deg)"], label="Latitude (deg)")
    ax2.plot(file_data["Timestamp"], file_data["Longitude (deg)"], label="Longitude (deg)")
    ax2.plot(file_data["Timestamp"], file_data["Altitude (km)"], label="Altitude (km)")
    ax2.set_ylabel("LLA Values")
    ax2.set_xlabel("Timestamp")
    ax2.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------------------------
# 8. Use the Plotting Function
# ---------------------------
unique_test_file_ids = np.unique(file_ids_test)
logger.info("Test set file IDs: " + ", ".join(map(str, unique_test_file_ids)))
random_file_id = np.random.choice(unique_test_file_ids)
logger.info(f"Plotting predictions for File ID: {random_file_id}")
plot_file_predictions(
    random_file_id,
    data_raw,
    model,
    seq_len,
    X_mean,
    X_std,
    y_mean,
    y_std,
    epsilon,
    selected_feature_columns=selected_feature_columns,
    target_col=target_col,
    device=device
)
from IPython import embed; embed(); quit()