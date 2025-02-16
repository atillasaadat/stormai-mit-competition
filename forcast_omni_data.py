import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from tqdm import tqdm  # Ensure you have tqdm installed

# Set up logging with a default INFO level.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

#############################################
# 1. Define the PatchTST-inspired model
#############################################
class PatchTST(nn.Module):
    """
    A simplified PatchTST model.
    This model "patchifies" the input time series with a 1D convolution,
    processes the patches with Transformer encoder layers, and then outputs
    a forecast for a fixed horizon. The input has many features, but the final
    forecast is only for the target features.
    """
    def __init__(self, input_size, patch_size, d_model, n_heads, n_layers, 
                 forecast_horizon, target_size, dropout=0.1):
        """
        Args:
            input_size (int): Number of input features.
            patch_size (int): Number of timesteps per patch.
            d_model (int): Dimension of patch embedding.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of Transformer encoder layers.
            forecast_horizon (int): How many future timesteps to forecast.
            target_size (int): Number of target features to predict.
            dropout (float): Dropout rate.
        """
        super(PatchTST, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon  # store for reshaping
        self.target_size = target_size            # store for reshaping

        logger.debug("Initializing PatchTST model.")
        # Patchify the time series along the time axis.
        self.proj = nn.Conv1d(in_channels=input_size, out_channels=d_model,
                              kernel_size=patch_size, stride=patch_size)

        # Use batch_first=True for the Transformer layers.
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        # Forecast head outputs forecast_horizon steps for each target feature.
        self.fc = nn.Linear(d_model, forecast_horizon * target_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        Returns:
            out: Forecasted output of shape (batch, forecast_horizon, target_size)
        """
        batch_size = x.size(0)
        # Conv1d expects (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        x = self.proj(x)       # (batch, d_model, new_seq_len)
        x = x.transpose(1, 2)  # (batch, new_seq_len, d_model)
        # With batch_first=True, transformer expects (batch, seq_len, d_model)
        x = self.transformer_encoder(x)
        # Use the last patch's representation
        x_last = x[:, -1, :]
        out = self.fc(x_last)
        out = out.view(batch_size, self.forecast_horizon, self.target_size)
        return out

#############################################
# 2. Define a PyTorch dataset for sliding windows
#############################################
class TimeSeriesDataset(data.Dataset):
    """
    Creates sliding-window examples from the full time series.
    Each sample is a tuple (x, y) where:
      - x: a window of input features of length input_window.
      - y: the next forecast_horizon values for the target features.
    """
    def __init__(self, X, Y, input_window, forecast_horizon):
        """
        Args:
            X (np.array): Input features array of shape (n_timesteps, n_input_features)
            Y (np.array): Target features array of shape (n_timesteps, n_target_features)
            input_window (int): Length of the input window.
            forecast_horizon (int): Length of the forecast window.
        """
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

#############################################
# 3. Training, evaluation, and forecasting functions
#############################################
def train_model(model, train_loader, val_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device)
    logger.info("Starting training loop.")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f"Epoch {epoch+1}/{epochs} Training", unit="batch")
        for batch_idx, (x, y) in train_pbar:
            # Check for NaNs in inputs
            if torch.isnan(x).any():
                logger.error(f"Input batch contains NaNs. Batch {batch_idx} at epoch {epoch+1}")
            if torch.isnan(y).any():
                logger.error(f"Target batch contains NaNs. Batch {batch_idx} at epoch {epoch+1}")

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
            train_pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        train_loss /= len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")

        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                        desc=f"Epoch {epoch+1}/{epochs} Validation", unit="batch")
        with torch.no_grad():
            for batch_idx, (x, y) in val_pbar:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item() * x.size(0)
                val_pbar.set_postfix(loss=f"{loss.item():.6f}")
        val_loss /= len(val_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.6f}")

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test set and compute MSE and MAE.
    Returns:
        mse_avg, mae_avg, all_preds, all_trues
    """
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    total_elements = 0
    all_preds = []
    all_trues = []
    criterion = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_trues.append(y.cpu())
            mse_sum += criterion(preds, y).item()
            mae_sum += torch.sum(torch.abs(preds - y)).item()
            total_elements += y.numel()
    mse_avg = mse_sum / total_elements
    mae_avg = mae_sum / total_elements
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)
    logger.info(f"Evaluation complete: MSE={mse_avg:.6f}, MAE={mae_avg:.6f}")
    return mse_avg, mae_avg, all_preds, all_trues

def forecast(model, input_sequence, device):
    """
    Forecast the next forecast_horizon steps given an input sequence.
    Args:
        input_sequence: NumPy array of shape (input_window, n_input_features)
    Returns:
        output: NumPy array of shape (forecast_horizon, n_target_features)
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)
    return output.squeeze(0).cpu().numpy()

#############################################
# 4. Plotting functions for combined historical & forecast plots
#############################################
def plot_combined_forecast(history_timestamps, history_values, forecast_timestamps, forecast_values, 
                           ylabel, title, true_forecast=None):
    """
    Plot a combined time series showing historical data and forecasted values.
    Optionally, also plot true forecast values.
    Args:
        history_timestamps (array-like): Timestamps for historical data.
        history_values (array-like): Historical target values.
        forecast_timestamps (array-like): Timestamps for forecasted data.
        forecast_values (array-like): Forecasted target values.
        ylabel (str): Label for the Y-axis.
        title (str): Plot title.
        true_forecast (array-like, optional): True forecast values.
    """
    logger.debug("Plotting combined forecast.")
    plt.figure(figsize=(12, 6))
    plt.plot(history_timestamps, history_values, label="Historical", marker='o', color='blue')
    plt.plot(forecast_timestamps, forecast_values, label="Forecast", marker='x', color='red')
    if true_forecast is not None:
        plt.plot(forecast_timestamps, true_forecast, label="True Forecast", marker='o', linestyle='--', color='green')
    plt.xlabel("Timestamp")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

#############################################
# 5. DataHandler class (provided)
#############################################
class DataHandler():
    def __init__(self, omni_folder, initial_state_folder, sat_density_folder, forcasted_omni2_data_folder):
        self.omni_folder = omni_folder
        self.initial_state_folder = initial_state_folder
        self.sat_density_folder = sat_density_folder
        self.forcasted_omni2_data_folder = forcasted_omni2_data_folder
        self.__read_initial_states()

    def __read_initial_states(self):
        all_dataframes = []
        for file in self.initial_state_folder.iterdir():
            if file.suffix == '.csv':
                df = pd.read_csv(file)
                all_dataframes.append(df)
        self.initial_states = pd.concat(all_dataframes, ignore_index=True)
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

#############################################
# 6. Main training, evaluation, and forecasting pipeline
#############################################
def main():
    logger.info("Starting main pipeline.")
    # Set up folder paths (adjust as needed)
    omni_folder = Path("./data/omni2")
    initial_state_folder = Path("./data/initial_state")
    sat_density_folder = Path("./data/sat_density")
    forcasted_omni2_data_folder = Path("./data/forcasted_omni2")

    # Instantiate DataHandler.
    data_handler = DataHandler(omni_folder, initial_state_folder, sat_density_folder, forcasted_omni2_data_folder)

    # ---------------------------
    # Load all files based on initial_states File ID.
    # ---------------------------
    load_percentage = 0.01  # For testing, load only 10% of file IDs.
    file_ids = data_handler.initial_states["File ID"].unique()
    logger.info(f"Total file IDs available: {len(file_ids)}")
    if load_percentage < 1.0:
        file_ids = np.random.choice(file_ids, size=int(load_percentage * len(file_ids)), replace=False)
        logger.info(f"Randomly selected {len(file_ids)} file IDs ({load_percentage*100:.0f}%) for loading.")

    omni_dfs = []
    sat_density_dfs = []
    total_ids = len(file_ids)
    next_threshold = 0.05  # 5% increments

    for i, fid in enumerate(file_ids):
        try:
            omni_dfs.append(data_handler.read_omni_data(fid))
        except FileNotFoundError:
            print(f"OMNI file for File ID {fid} not found; skipping.")
        try:
            sat_density_dfs.append(data_handler.read_sat_density_data(fid))
        except FileNotFoundError:
            print(f"Sat density file for File ID {fid} not found; skipping.")

        progress = (i + 1) / total_ids
        if progress >= next_threshold:
            print(f"{int(next_threshold * 100)}% complete")
            next_threshold += 0.05

    if len(omni_dfs) == 0 or len(sat_density_dfs) == 0:
        raise ValueError("No valid OMNI or sat_density files were loaded.")
    omni_df = pd.concat(omni_dfs, ignore_index=True)
    sat_density_df = pd.concat(sat_density_dfs, ignore_index=True)
    logger.info(f"Loaded {len(omni_df)} rows of OMNI data and {len(sat_density_df)} rows of sat density data.")

    # Preprocess: convert Timestamps and sort.
    omni_df['Timestamp'] = pd.to_datetime(omni_df['Timestamp'])
    omni_df.sort_values('Timestamp', inplace=True)
    sat_density_df['Timestamp'] = pd.to_datetime(sat_density_df['Timestamp'])
    sat_density_df.sort_values('Timestamp', inplace=True)
    logger.debug("Timestamps converted and sorted.")

    # ---------------------------
    # Define input and target features.
    # Only select suggested features.
    # Suggested for "ap_index_nT":
    ap_features = [
        "Kp_index", "Dst_index_nT", "AU_index_nT", "AL_index_nT", "AE_index_nT",
        "SW_Plasma_Speed_km_s", "SW_Proton_Density_N_cm3", "SW_Plasma_Temperature_K",
        "Scalar_B_nT", "Vector_B_Magnitude_nT", "BX_nT_GSE_GSM", "BY_nT_GSE", "BZ_nT_GSE",
        "Plasma_Beta", "Flow_pressure"
    ]
    # Suggested for "f10.7_index":
    f107_features = [
        "Bartels_rotation_number", "R_Sunspot_No", "Lyman_alpha"
    ]
    candidate_features = list(set(ap_features).union(set(f107_features)))
    candidate_features = [f for f in candidate_features if f in omni_df.columns]

    # Remove features that are all NaN or constant.
    cleaned_features = []
    for f in candidate_features:
        if omni_df[f].isna().all():
            logger.debug(f"Feature {f} is all NaN, removing.")
            continue
        if omni_df[f].nunique() <= 1:
            logger.debug(f"Feature {f} is constant, removing.")
            continue
        cleaned_features.append(f)

    input_features = cleaned_features
    target_features = ["ap_index_nT", "f10.7_index"]

    logger.info(f"Selected input features: {input_features}")
    logger.info(f"Target features: {target_features}")

    # Fill any remaining NaN values in input features with their median.
    for col in input_features:
        if omni_df[col].isna().any():
            median_val = omni_df[col].median()
            logger.debug(f"Filling NaNs in {col} with median value {median_val}.")
            omni_df[col] = omni_df[col].fillna(median_val)

    # Normalize the input features.
    input_data = omni_df[input_features]
    input_data = (input_data - input_data.mean()) / input_data.std()
    omni_df[input_features] = input_data

    # Create input (X) and target (Y) arrays.
    X = omni_df[input_features].values  # shape: (n_timesteps, n_input_features)
    Y = omni_df[target_features].values   # shape: (n_timesteps, n_target_features)
    logger.info(f"Input features shape: {X.shape}, Target features shape: {Y.shape}")

    # ---------------------------
    # Define forecasting parameters.
    # ---------------------------
    input_window = 100  # e.g., 100 timesteps of historical data
    unique_timestamps = sat_density_df["Timestamp"].drop_duplicates().reset_index(drop=True)
    forecast_horizon = len(unique_timestamps)  # one prediction per unique timestamp
    logger.info(f"Using input_window={input_window} and forecast_horizon={forecast_horizon}")

    # ---------------------------
    # Train–Test (80/20) split.
    # ---------------------------
    split_idx = int(0.8 * len(X))
    logger.info(f"Train/Test split at index {split_idx} out of {len(X)} samples.")
    train_X = X[:split_idx]
    train_Y = Y[:split_idx]
    test_X = X[split_idx - input_window - forecast_horizon + 1:]
    test_Y = Y[split_idx - input_window - forecast_horizon + 1:]

    # Create datasets and loaders.
    batch_size = 32
    train_dataset = TimeSeriesDataset(train_X, train_Y, input_window, forecast_horizon)
    test_dataset = TimeSeriesDataset(test_X, test_Y, input_window, forecast_horizon)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")

    # ---------------------------
    # Define and instantiate the model.
    # ---------------------------
    patch_size = 10    # timesteps per patch
    d_model = 64
    n_heads = 4
    n_layers = 2
    dropout = 0.1
    model = PatchTST(input_size=len(input_features), patch_size=patch_size, 
                     d_model=d_model, n_heads=n_heads, n_layers=n_layers, 
                     forecast_horizon=forecast_horizon, target_size=len(target_features), 
                     dropout=dropout)
    logger.debug("Model instantiated.")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info("Cuda available: %s", torch.cuda.is_available())
    else:
        raise ValueError("Cuda not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10  # Increase for full training.
    lr = 1e-3

    logger.info("Starting training...")
    train_model(model, train_loader, test_loader, epochs, lr, device)

    # ---------------------------
    # Evaluate the model on the test set and print metrics.
    # ---------------------------
    test_mse, test_mae, preds, trues = evaluate_model(model, test_loader, device)
    logger.info(f"Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")

    # ---------------------------
    # Forecast using the last input_window from the test set.
    # Map the test sample back to the original omni_df indices.
    # ---------------------------
    test_offset = split_idx - input_window - forecast_horizon + 1
    sample_index = 0  # choose the first test sample (change as needed)
    global_index = test_offset + sample_index
    logger.debug(f"Using global index {global_index} for forecast sample.")

    # Extract historical data and true forecast values, converting to NumPy arrays.
    history_timestamps = omni_df['Timestamp'].iloc[global_index : global_index + input_window].values
    history_ap = omni_df['ap_index_nT'].iloc[global_index : global_index + input_window].values
    history_f107 = omni_df['f10.7_index'].iloc[global_index : global_index + input_window].values

    forecast_timestamps = omni_df['Timestamp'].iloc[global_index + input_window : global_index + input_window + forecast_horizon].values
    true_forecast_ap = omni_df['ap_index_nT'].iloc[global_index + input_window : global_index + input_window + forecast_horizon].values
    true_forecast_f107 = omni_df['f10.7_index'].iloc[global_index + input_window : global_index + input_window + forecast_horizon].values

    sample_x, _ = test_dataset[sample_index]
    sample_x = sample_x.unsqueeze(0).to(device)
    pred_sample = model(sample_x).cpu().numpy().squeeze(0)  # shape: (forecast_horizon, target_size)
    pred_ap = pred_sample[:, 0]
    pred_f107 = pred_sample[:, 1]
    logger.info("Forecast sample computed.")

    # ---------------------------
    # Plot combined timeseries for ap_index_nT.
    # ---------------------------
    plot_combined_forecast(history_timestamps, history_ap, forecast_timestamps, pred_ap,
                           ylabel="ap_index_nT", title="Historical and Forecasted ap_index_nT",
                           true_forecast=true_forecast_ap)
    # ---------------------------
    # Plot combined timeseries for f10.7_index.
    # ---------------------------
    plot_combined_forecast(history_timestamps, history_f107, forecast_timestamps, pred_f107,
                           ylabel="f10.7_index", title="Historical and Forecasted f10.7_index",
                           true_forecast=true_forecast_f107)
    logger.info("Plots generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and forecast space weather indices with PatchTST.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled.")
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Debug logging disabled; using INFO level logging.")

    main()
