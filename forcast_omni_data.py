import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt

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

        # Patchify the time series along the time axis.
        self.proj = nn.Conv1d(in_channels=input_size, out_channels=d_model,
                              kernel_size=patch_size, stride=patch_size)

        # Transformer encoder layers.
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout)
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
        x = self.proj(x)  # -> (batch, d_model, new_seq_len)
        x = x.transpose(1, 2)  # -> (batch, new_seq_len, d_model)
        # Transformer expects (sequence, batch, features)
        x = x.transpose(0, 1)  # -> (new_seq_len, batch, d_model)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # -> (batch, new_seq_len, d_model)
        # Use the last patch’s representation as a summary.
        x_last = x[:, -1, :]  # -> (batch, d_model)
        out = self.fc(x_last)  # -> (batch, forecast_horizon * target_size)
        # Reshape to (batch, forecast_horizon, target_size)
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.input_window]
        y = self.Y[idx + self.input_window : idx + self.input_window + self.forecast_horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

#############################################
# 3. Training and forecasting functions
#############################################
def train_model(model, train_loader, val_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.6f}")

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
# 4. Plotting functions for model performance
#############################################
def plot_forecast_ap_index(pred, actual, title="Forecast vs Actual for ap_index_nT"):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual ap_index_nT", marker='o')
    plt.plot(pred, label="Predicted ap_index_nT", marker='x')
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("ap_index_nT")
    plt.legend()
    plt.show()

def plot_forecast_f107_index(pred, actual, title="Forecast vs Actual for f10.7_index"):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual f10.7_index", marker='o')
    plt.plot(pred, label="Predicted f10.7_index", marker='x')
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("f10.7_index")
    plt.legend()
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

    def read_omni_data(self, file_id):
        file_id_str = f"{file_id:05d}"
        for file in self.omni_folder.iterdir():
            if file.suffix == '.csv' and file_id_str in file.stem:
                return pd.read_csv(file)
        raise FileNotFoundError(f"File with ID {file_id} not found in {self.omni_folder}")

    def read_sat_density_data(self, file_id):
        file_id_str = f"{file_id:05d}"
        for file in self.sat_density_folder.iterdir():
            if file.suffix == '.csv' and file_id_str in file.stem:
                return pd.read_csv(file)
        raise FileNotFoundError(f"File with ID {file_id} not found in {self.sat_density_folder}")

#############################################
# 6. Main training and forecasting pipeline
#############################################
def main():
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
    file_ids = data_handler.initial_states["File ID"].unique()
    omni_dfs = []
    sat_density_dfs = []
    for fid in file_ids:
        try:
            omni_dfs.append(data_handler.read_omni_data(fid))
        except FileNotFoundError:
            print(f"OMNI file for File ID {fid} not found; skipping.")
        try:
            sat_density_dfs.append(data_handler.read_sat_density_data(fid))
        except FileNotFoundError:
            print(f"Sat density file for File ID {fid} not found; skipping.")
    if len(omni_dfs) == 0 or len(sat_density_dfs) == 0:
        raise ValueError("No valid OMNI or sat_density files were loaded.")
    omni_df = pd.concat(omni_dfs, ignore_index=True)
    sat_density_df = pd.concat(sat_density_dfs, ignore_index=True)

    # Preprocess: convert Timestamps and sort.
    omni_df['Timestamp'] = pd.to_datetime(omni_df['Timestamp'])
    omni_df.sort_values('Timestamp', inplace=True)
    sat_density_df['Timestamp'] = pd.to_datetime(sat_density_df['Timestamp'])
    sat_density_df.sort_values('Timestamp', inplace=True)

    # ---------------------------
    # Define input and target features.
    # Use all numeric columns (except Timestamp) as input;
    # only predict ap_index_nT and f10.7_index.
    # ---------------------------
    # (Adjust this list if you want to filter out undesired features.)
    all_features = [col for col in omni_df.columns if col != 'Timestamp']
    target_features = ["ap_index_nT", "f10.7_index"]
    # Option 1: Use all features (including targets) as input.
    # Option 2: Use only non-target columns as input.
    # Here we choose Option 2 to emphasize the influence of the other columns.
    input_features = [f for f in all_features if f not in target_features]

    # Create input (X) and target (Y) arrays.
    X = omni_df[input_features].values  # shape: (n_timesteps, n_input_features)
    Y = omni_df[target_features].values  # shape: (n_timesteps, n_target_features)

    # ---------------------------
    # Define forecasting parameters.
    # ---------------------------
    input_window = 100                    # number of past timesteps to use
    forecast_horizon = len(sat_density_df) # forecast horizon set to # of sat_density timestamps

    # Split into training and validation sets (80/20 split).
    split_idx = int(0.8 * len(X))
    train_X = X[:split_idx]
    train_Y = Y[:split_idx]
    # Ensure the validation set has enough context.
    val_X = X[split_idx - input_window - forecast_horizon + 1:]
    val_Y = Y[split_idx - input_window - forecast_horizon + 1:]

    # Create datasets and loaders.
    train_dataset = TimeSeriesDataset(train_X, train_Y, input_window, forecast_horizon)
    val_dataset = TimeSeriesDataset(val_X, val_Y, input_window, forecast_horizon)
    batch_size = 32
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---------------------------
    # Define and instantiate the model.
    # ---------------------------
    # input_size: number of input features,
    # target_size: number of target features (2 in our case).
    patch_size = 10    # timesteps per patch
    d_model = 64
    n_heads = 4
    n_layers = 2
    dropout = 0.1
    model = PatchTST(input_size=len(input_features), patch_size=patch_size, 
                     d_model=d_model, n_heads=n_heads, n_layers=n_layers, 
                     forecast_horizon=forecast_horizon, target_size=len(target_features), 
                     dropout=dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10  # Increase for full training.
    lr = 1e-3

    print("Starting training...")
    train_model(model, train_loader, val_loader, epochs, lr, device)

    # ---------------------------
    # Forecast using the last input_window from the full series.
    # Also extract the corresponding ground truth targets.
    # ---------------------------
    input_sequence = X[-input_window:]
    forecasted_values = forecast(model, input_sequence, device)  # shape: (forecast_horizon, n_target_features)
    true_values = Y[-forecast_horizon:]  # ground truth for target features

    # Save the forecasted data (aligned to sat_density timestamps).
    forecast_df = pd.DataFrame(forecasted_values, columns=target_features)
    forecast_df["Timestamp"] = sat_density_df["Timestamp"].values[:forecast_horizon]
    forecast_file = forcasted_omni2_data_folder / "forecasted_omni.csv"
    forecast_df.to_csv(forecast_file, index=False)
    print(f"Forecasted data saved to {forecast_file}")

    # ---------------------------
    # Plotting results (using a subset of the forecast horizon for clarity).
    # ---------------------------
    plot_horizon = min(100, forecast_horizon)
    # For ap_index_nT (target column index 0)
    pred_ap = forecasted_values[:plot_horizon, 0]
    true_ap = true_values[:plot_horizon, 0]
    plot_forecast_ap_index(pred_ap, true_ap)
    # For f10.7_index (target column index 1)
    pred_f107 = forecasted_values[:plot_horizon, 1]
    true_f107 = true_values[:plot_horizon, 1]
    plot_forecast_f107_index(pred_f107, true_f107)

if __name__ == "__main__":
    main()
