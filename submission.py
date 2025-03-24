"""
Combined simulation propagation and model prediction pipeline.

This script:
  1. Loads the space weather (OMNI2) data and initial state.
  2. Uses Orekit propagation (via SimulationRunner) to generate a dataframe
     that includes the computed MSIS density and geolocation information.
  3. Prepares the propagated dataframe by computing additional features (e.g. log_Altitude).
  4. Loads a pre-trained PyTorch model (saved in a .pth file) and uses it to predict
     the true orbit mean density using a sliding window over the propagated features.
  5. Runs the pipeline for every file_id found in the initial state data in parallel,
     using all available CPUs.
  6. Outputs a JSON object with file IDs as keys and a dictionary containing ISO‐format
     Timestamps and predicted “Orbit Mean Density (kg/m³)” values to /app/output/prediction.json.
     
Minimal changes have been made to the original code to integrate both pipelines.
"""

import os
import math
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import multiprocessing

# Import the DataHandler and SimulationRunner classes from your modules.
from datahandler import DataHandler
from orekit_propagator import SimulationRunner

# ---------------------------
# Model Definition (same as used in training)
# ---------------------------
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
        # Reshape input into patches
        x = x.reshape(B, self.num_patches, self.patch_size * C)
        x = self.proj(x)
        x = self.transformer_encoder(x)
        # Use the last patch's output for prediction
        x = x[:, -1, :]
        out = self.fc(x)
        return out

# ---------------------------
# Prediction function using a dataframe
# ---------------------------
def predict_from_dataframe(df, model, seq_len, selected_feature_columns, target_factor=1e12, epsilon=1e-8):
    """
    Given a dataframe with propagated results, this function:
      - Computes log_Altitude if missing.
      - Prepares the input features (scaling them based on the data).
      - Runs a sliding-window prediction using the loaded model.
      - Returns two lists: ISO-format timestamps and predicted orbit densities.
    """
    # Ensure Timestamp is datetime and sort
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Compute log_Altitude from "Altitude (km)" if not already there.
    if "log_Altitude" not in df.columns:
        df["log_Altitude"] = np.log10(df["Altitude (km)"])

    # Build feature matrix from selected columns.
    X_input = df[selected_feature_columns].values

    # For prediction, compute scaling parameters from the current file.
    X_mean = np.mean(X_input, axis=0)
    X_std = np.std(X_input, axis=0)
    X_scaled = (X_input - X_mean) / (X_std + epsilon)

    # In this prediction-only phase, assume target scaling parameters are defaults:
    y_mean = np.array([0.0])
    y_std = np.array([1.0])
    
    predictions = []
    pred_timestamps = []
    num_windows = len(X_scaled) - seq_len
    device = next(model.parameters()).device

    for i in range(num_windows):
        window = X_scaled[i:i+seq_len]  # shape: (seq_len, num_features)
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)  # (1, seq_len, num_features)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(window_tensor)
        # Inverse scaling (using provided defaults)
        pred_value = (pred_scaled.cpu().item() * y_std[0] + y_mean[0]) / target_factor
        predictions.append(pred_value)
        
        # Get the timestamp and ensure it is timezone-aware (set to UTC if naive)
        ts = df.iloc[i+seq_len]["Timestamp"]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        pred_timestamps.append(ts.isoformat())
    
    return pred_timestamps, predictions

# ---------------------------
# Worker function for processing one file_id
# ---------------------------
def process_file(file_id):
    """
    Processes one file_id: runs propagation simulation and prediction.
    Returns a dictionary: {str(file_id): { "Timestamp": [...], "Orbit Mean Density (kg/m^3)": [...] }}
    """
    # Reinitialize Orekit JVM in this process:

    # Set up a logger (each process can have its own)
    logger = logging.getLogger(f"Process-{file_id}")
    logger.setLevel(logging.INFO)

    # Define DATA_PATHS and parameters (using the absolute paths required by the submission)
    DATA_PATHS = {
        "omni2_folder": Path("/app/data/dataset/test/omni2"),
        "initial_state_file": Path("/app/input_data/initial_states.csv"),
        #"sat_density_folder": Path("/app/data/dataset/test/sat_density"),
        "sat_density_folder": None,
        "forcasted_omni2_folder": Path("/app/data/dataset/test/forcasted_omni2"),
        "sat_density_omni_forcasted_folder": Path("/app/data/dataset/test/sat_density_omni_forcasted"),
        "sat_density_omni_propagated_folder": Path("/app/data/dataset/test/sat_density_omni_propagated"),
    }
    # Propagation simulation parameters
    sat_config = {
        "satellite_mass_kg": 100.0,
        "cross_section_m2": 1.0,
        "srp_area_m2": 1.0,
        "drag_coeff": 2.2,
        "cr": 1.0,
    }
    sim_config = {
        "min_step": 1e-6,
        "max_step": 100.0,
        "init_step": 5.0,
        "pos_tol": 1e-3,
        "spherical_harmonics": (4, 4),
    }
    # Prediction parameters
    SEQ_LEN = 10
    PATCH_SIZE = 2
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 3
    SELECTED_FEATURE_COLUMNS = ["MSIS Density (kg/m^3)", "Latitude (deg)", "Longitude (deg)", "log_Altitude"]
    TARGET_FACTOR = 1e12
    EPSILON = 1e-8
    MODEL_PATH = Path("density_prediction_patchtst_model.pth")
    
    try:
        # Instantiate DataHandler and SimulationRunner for this process
        dh = DataHandler(logger, **DATA_PATHS)
        sim_runner = SimulationRunner(logger, sat_config, sim_config)
        
        logger.info(f"Processing file ID {file_id:05d}")
        # Load required input data for this file_id
        initial_state = dh.get_initial_state(file_id)
        omni_data = dh.read_csv_data(file_id, dh.omni2_folder)
        sat_density_truth = None
        if dh.sat_density_folder is not None:
            sat_density_truth = dh.read_csv_data(file_id, dh.sat_density_folder)
        
        # Run orbit propagation simulation
        propagated_df = sim_runner.run_simulation(
            file_id,
            initial_state=initial_state,
            space_weather_data=omni_data,
            sat_density_truth=sat_density_truth
        )
        logger.info(f"Propagation complete for file {file_id:05d} with {len(propagated_df)} timesteps.")
        
        # Ensure the dataframe has a "log_Altitude" column.
        if "log_Altitude" not in propagated_df.columns:
            propagated_df["log_Altitude"] = np.log10(propagated_df["Altitude (km)"])
        
        # Load the trained model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_features = len(SELECTED_FEATURE_COLUMNS)
        model = PatchTST(seq_len=SEQ_LEN, patch_size=PATCH_SIZE, num_features=num_features,
                         d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS)
        model.to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        # Run predictions using a sliding window over the propagated dataframe.
        timestamps, predicted_densities = predict_from_dataframe(
            propagated_df, model, SEQ_LEN, SELECTED_FEATURE_COLUMNS, TARGET_FACTOR, EPSILON
        )
        
        result = {
            "Timestamp": timestamps,
            "Orbit Mean Density (kg/m^3)": predicted_densities,
        }
        logger.info(f"Prediction complete for file {file_id:05d}")
        return {str(file_id): result}
    except Exception as e:
        logger.error(f"Error processing file ID {file_id}: {e}")
        return {str(file_id): {"error": str(e)}}

# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    # Set multiprocessing start method to spawn.
    multiprocessing.set_start_method("spawn", force=True)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Main")
    
    # Define DATA_PATHS (same as in worker) to instantiate DataHandler
    DATA_PATHS = {
        "omni2_folder": Path("/app/data/dataset/test/omni2"),
        "initial_state_file": Path("/app/input_data/initial_states.csv"),
        #"sat_density_folder": Path("/app/data/dataset/test/sat_density"),
        "sat_density_folder": None,
        "forcasted_omni2_folder": Path("/app/data/dataset/test/forcasted_omni2"),
        "sat_density_omni_forcasted_folder": Path("/app/data/dataset/test/sat_density_omni_forcasted"),
        "sat_density_omni_propagated_folder": Path("/app/data/dataset/test/sat_density_omni_propagated"),
    }
    
    # Instantiate DataHandler in main to retrieve all file IDs from the initial states.
    dh = DataHandler(logger, **DATA_PATHS)
    # Assume the initial states CSV has a column "File ID" with the file IDs.
    file_ids = dh.initial_states["File ID"].unique().tolist()
    logger.info(f"Found {len(file_ids)} file IDs in the initial states.")

    import os
    os.environ["MSIS_DATA_DIR"] = "/app/ingested_program/src/msis2.0"
    
    import orekit
    from orekit.pyhelpers import setup_orekit_curdir
    orekit.initVM()
    setup_orekit_curdir(filename="orekit-data.zip", from_pip_library=False)

    # Use multiprocessing Pool to process each file_id in parallel.
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"Using {cpu_count} CPUs for processing.")
    with multiprocessing.Pool(processes=cpu_count) as pool:
        results = pool.map(process_file, file_ids)
    
    # Combine results from all processes
    output_dict = {}
    for res in results:
        output_dict.update(res)
    
    # Save the combined JSON output.
    json_output = json.dumps(output_dict, indent=4)
    with open("/app/output/prediction.json", "w") as f:
        f.write(json_output)
    logger.info("Prediction JSON saved to /app/output/prediction.json")

if __name__ == "__main__":
    main()
