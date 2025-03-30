"""
Combined simulation propagation and model prediction pipeline.

This script:
  1. Loads the space weather (OMNI2) data and initial state.
  2. Uses our J₂/pyMSIS propagation to generate a dataframe that includes computed MSIS density and geolocation info.
  3. Prepares the propagated dataframe by computing additional features (e.g. log_Altitude).
  4. Loads a pre-trained PyTorch model (saved in a .pth file) and uses it to predict the correction ratio
     (true density / MSIS density) over a sliding window. The predicted ratio is multiplied by the MSIS density
     to yield the corrected density.
  5. Runs the pipeline for every file_id in parallel (or sequentially) and outputs a JSON object
     with file IDs as keys and a dictionary containing ISO‐format Timestamps and predicted “Orbit Mean Density (kg/m³)” values.
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
import time
import argparse
from functools import partial
from datetime import datetime, timedelta

# Import propagation functions and dependencies.
from pymsis import msis
from pyproj import Transformer
from numba import njit
from datahandler import DataHandler

# ---------------------------
# J2/pyMSIS Propagation Functions
# ---------------------------

# Earth's gravitational parameter and WGS84 equatorial radius.
MU = 3.986004418e14       # [m^3/s^2]
R_E = 6378137.0           # [m]

# Transformer from ECEF (EPSG:4978) to WGS84 geodetic (EPSG:4326)
ecef_to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

def oe2rv(a_km, e, i_deg, raan_deg, argp_deg, nu_deg):
    """
    Convert orbital elements (in km, deg) to ECI state vectors (m, m/s).
    """
    i = math.radians(i_deg)
    raan = math.radians(raan_deg)
    argp = math.radians(argp_deg)
    nu = math.radians(nu_deg)
    
    a = a_km * 1e3
    p = a * (1 - e**2)
    r_val = p / (1 + e * math.cos(nu))
    
    r_pf = np.array([r_val * math.cos(nu), r_val * math.sin(nu), 0.0])
    v_pf = np.array([
        -math.sqrt(MU / p) * math.sin(nu),
         math.sqrt(MU / p) * (e + math.cos(nu)),
         0.0
    ])
    
    R = np.array([
        [math.cos(raan)*math.cos(argp) - math.sin(raan)*math.sin(argp)*math.cos(i),
         -math.cos(raan)*math.sin(argp) - math.sin(raan)*math.cos(argp)*math.cos(i),
         math.sin(raan)*math.sin(i)],
        [math.sin(raan)*math.cos(argp) + math.cos(raan)*math.sin(argp)*math.cos(i),
         -math.sin(raan)*math.sin(argp) + math.cos(raan)*math.cos(argp)*math.cos(i),
         -math.cos(raan)*math.sin(i)],
        [math.sin(argp)*math.sin(i),
         math.cos(argp)*math.sin(i),
         math.cos(i)]
    ])
    
    r_eci = R @ r_pf
    v_eci = R @ v_pf
    return r_eci, v_eci

def gmst(dt):
    """
    Compute an approximate Greenwich Mean Sidereal Time (radians)
    for a given datetime (UTC).
    """
    JD = (dt - datetime(2000, 1, 1, 12)).total_seconds()/86400.0 + 2451545.0
    GMST_hours = 18.697374558 + 24.06570982441908 * (JD - 2451545.0)
    GMST_hours %= 24.0
    return (GMST_hours / 24.0) * 2 * math.pi

def eci_to_ecef(r_eci, current_time):
    """
    Convert an ECI position vector (m) to ECEF coordinates using GMST rotation.
    """
    theta = gmst(current_time)
    cos_theta = math.cos(-theta)
    sin_theta = math.sin(-theta)
    R = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta,  cos_theta, 0],
                  [0,          0,         1]])
    return R @ r_eci

def get_density(current_time, r_eci):
    """
    Convert the current ECI position to geodetic coordinates (via ECEF),
    then call pymsis to obtain the atmospheric density.
    MSIS expects altitude in km.
    """
    r_ecef = eci_to_ecef(r_eci, current_time)
    x, y, z = r_ecef
    lon, lat, alt_m = ecef_to_geo.transform(x, y, z)
    alt_km = alt_m / 1000.0
    f107_daily = 155.1
    aps = [15] * 7
    try:
        result = msis.run(
            dates=[current_time],
            lons=[lon],
            lats=[lat],
            alts=[alt_km],
            f107s=[f107_daily],
            aps=[aps]
        )
        density = result[0, 0]
    except Exception as e:
        raise Exception(f"Error running MSIS at alt {alt_km:.2f} km: {e}")
    return density

@njit
def gravitational_acceleration(r):
    x, y, z = r[0], r[1], r[2]
    r_norm = math.sqrt(x*x + y*y + z*z)
    r_norm3 = r_norm**3
    acc = np.array([-MU*x / r_norm3,
                    -MU*y / r_norm3,
                    -MU*z / r_norm3])
    J2 = 1.08263e-3
    factor = 1.5 * J2 * MU * (R_E**2) / (r_norm**5)
    acc[0] += factor * x * (5*(z**2)/(r_norm**2) - 1)
    acc[1] += factor * y * (5*(z**2)/(r_norm**2) - 1)
    acc[2] += factor * z * (5*(z**2)/(r_norm**2) - 3)
    return acc

@njit
def drag_from_density(density, v, mass, Cd, A):
    v_norm = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if v_norm < 1e-8:
        return np.zeros(3)
    return -0.5 * Cd * A * density * v_norm * v / mass

@njit
def rk4_step(r, v, density, dt, mass, Cd, A):
    k1_r = v
    k1_v = gravitational_acceleration(r) + drag_from_density(density, v, mass, Cd, A)
    
    r_temp = r + 0.5 * dt * k1_r
    v_temp = v + 0.5 * dt * k1_v
    k2_r = v_temp
    k2_v = gravitational_acceleration(r_temp) + drag_from_density(density, v_temp, mass, Cd, A)
    
    r_temp = r + 0.5 * dt * k2_r
    v_temp = v + 0.5 * dt * k2_v
    k3_r = v_temp
    k3_v = gravitational_acceleration(r_temp) + drag_from_density(density, v_temp, mass, Cd, A)
    
    r_temp = r + dt * k3_r
    v_temp = v + dt * k3_v
    k4_r = v_temp
    k4_v = gravitational_acceleration(r_temp) + drag_from_density(density, v_temp, mass, Cd, A)
    
    r_new = r + (dt/6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_new = v + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return r_new, v_new

def propagate_orbit(r0, v0, t0, dt, steps, mass, Cd, A):
    """
    Propagate the orbit from the initial state (r0, v0) starting at time t0 using RK4 integration.
    For each step, the dynamic MSIS density is computed (assumed constant over dt) and used in a jitted RK4 step.
    Returns lists of timestamps, state vectors, and densities.
    """
    times = [t0]
    states = [np.hstack((r0, v0))]
    densities = [get_density(t0, r0)]
    
    r = r0.copy()
    v = v0.copy()
    current_time = t0
    for i in range(steps):
        density = get_density(current_time, r)
        r, v = rk4_step(r, v, density, dt, mass, Cd, A)
        current_time = current_time + timedelta(seconds=dt)
        times.append(current_time)
        states.append(np.hstack((r, v)))
        densities.append(get_density(current_time, r))
    return times, np.array(states), densities

def eci_to_lla(r_eci, current_time):
    """
    Convert an ECI state vector to geodetic latitude, longitude, and altitude (km).
    """
    r_ecef = eci_to_ecef(r_eci, current_time)
    x, y, z = r_ecef
    lon, lat, alt_m = ecef_to_geo.transform(x, y, z)
    return lat, lon, alt_m/1000.0

# ---------------------------
# Model Definition (as used in training)
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
        x = x.reshape(B, self.num_patches, self.patch_size * C)
        x = self.proj(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return torch.relu(out)

# ---------------------------
# Prediction function using a dataframe
# ---------------------------
def predict_from_dataframe(df, model, seq_len, selected_feature_columns, epsilon=1e-8, scaling_params=None):
    """
    Given a dataframe with propagated results, this function:
      - Ensures Timestamps are datetime.
      - Computes log_Altitude if missing.
      - Scales the input features using saved scaling parameters.
      - Runs a sliding-window prediction using the loaded model.
      - For each window, multiplies the predicted ratio by the corresponding MSIS density.
      - Returns two lists: ISO-format timestamps and predicted orbit densities.
    """
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    if "log_Altitude" not in df.columns:
        df["log_Altitude"] = np.log10(df["Altitude (km)"])
    X_input = df[selected_feature_columns].values
    if scaling_params is None:
        X_mean = np.mean(X_input, axis=0)
        X_std = np.std(X_input, axis=0)
    else:
        X_mean = np.array(scaling_params["X_mean"])
        X_std = np.array(scaling_params["X_std"])
    X_scaled = (X_input - X_mean) / (X_std + epsilon)
    
    predictions = []
    pred_timestamps = []
    num_windows = len(X_scaled) - seq_len
    device = next(model.parameters()).device

    for i in range(num_windows):
        window = X_scaled[i:i+seq_len]
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            pred_ratio = model(window_tensor).cpu().item()
        msis_val = df.iloc[i+seq_len]["MSIS Density (kg/m^3)"]
        pred_density = pred_ratio * msis_val
        predictions.append(pred_density)
        ts = df.iloc[i+seq_len]["Timestamp"]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        pred_timestamps.append(ts.isoformat())
    
    return pred_timestamps, predictions

# ---------------------------
# Per-file processing function
# ---------------------------
def process_file(file_id, dh, model, seq_len, selected_feature_columns, epsilon, scaling_params, propagation_params):
    """
    Process a single file by:
      - Loading the initial state.
      - Using our J₂/pyMSIS propagation to create a propagated dataframe.
      - Running the prediction function.
      - Returning a dictionary with file_id as key and a dict with Timestamps and predicted densities.
    """
    try:
        logger = logging.getLogger(__name__)
        file_id_str = f"{file_id:05d}"
        # Load initial state.
        initial_state = dh.get_initial_state(file_id)
        t0 = pd.to_datetime(initial_state["Timestamp"])
        a_km = initial_state["Semi-major Axis (km)"]
        e = initial_state["Eccentricity"]
        i_deg = initial_state["Inclination (deg)"]
        raan_deg = initial_state["RAAN (deg)"]
        argp_deg = initial_state["Argument of Perigee (deg)"]
        nu_deg = initial_state["True Anomaly (deg)"]
        
        # Convert orbital elements to initial state vectors.
        r0, v0 = oe2rv(a_km, e, i_deg, raan_deg, argp_deg, nu_deg)
        
        # Propagation parameters.
        dt_integ = propagation_params["dt_integ"]
        steps_integ = propagation_params["steps_integ"]
        mass = propagation_params["mass"]
        Cd = propagation_params["Cd"]
        A = propagation_params["A"]
        
        # Propagate with our J₂/pyMSIS integrator.
        times_fine, states_fine, densities_fine = propagate_orbit(r0, v0, t0, dt_integ, steps_integ, mass, Cd, A)
        sample_rate = propagation_params["sample_rate"]
        times_prop = times_fine[::sample_rate]
        states_prop = states_fine[::sample_rate]
        densities_prop = [densities_fine[i] for i in range(0, len(densities_fine), sample_rate)]
        
        # Build a DataFrame with geodetic info.
        rows = []
        for t, state, dens in zip(times_prop, states_prop, densities_prop):
            r = state[0:3]
            lat_val, lon_val, alt_val = eci_to_lla(r, t)
            rows.append({
                "Timestamp": t,
                "Altitude (km)": alt_val,
                "MSIS Density (kg/m^3)": dens,
                "Latitude (deg)": lat_val,
                "Longitude (deg)": lon_val,
            })
        propagated_df = pd.DataFrame(rows)
        if "log_Altitude" not in propagated_df.columns:
            propagated_df["log_Altitude"] = np.log10(propagated_df["Altitude (km)"])
        
        # Run predictions using a sliding window over the propagated dataframe.
        pred_timestamps, predicted_densities = predict_from_dataframe(
            propagated_df, model, seq_len, selected_feature_columns, epsilon, scaling_params
        )

        # Convert predicted densities to python float type.
        predicted_densities = [float(d) for d in predicted_densities]

        logger.info(f"Prediction complete for file {file_id:05d}")
        return {str(file_id): {"Timestamp": pred_timestamps, "Orbit Mean Density (kg/m^3)": predicted_densities}}

    except Exception as e:
        logger.error(f"Error processing file ID {file_id}: {e}")
        return {str(file_id): {"error": str(e)}}

# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run simulation propagation and model prediction pipeline."
    )
    parser.add_argument("--single_process", action="store_true",
                        help="Enable run sequentially, otherwise multiprocessing")
    args = parser.parse_args()
    
    os.environ["MSIS_DATA_DIR"] = "/app/ingested_program/src/msis2.0"
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Main")
    
    # Define data paths.
    DATA_PATHS = {
        "omni2_folder": Path("/app/data/dataset/test/omni2"),
        "initial_state_file": Path("/app/input_data/initial_states.csv"),
        "sat_density_folder": None,
        "forcasted_omni2_folder": Path("/app/data/dataset/test/forcasted_omni2"),
        "sat_density_omni_forcasted_folder": Path("/app/data/dataset/test/sat_density_omni_forcasted"),
        "sat_density_omni_propagated_folder": Path("/app/data/dataset/test/sat_density_omni_propagated"),
    }
    
    # Instantiate DataHandler to obtain file IDs.
    dh = DataHandler(logger, **DATA_PATHS)
    file_ids = dh.initial_states["File ID"].unique().tolist()
    total_files = len(file_ids)
    logger.info(f"Found {total_files} file IDs to process.")
    
    # Set prediction parameters.
    seq_len = 10
    selected_feature_columns = ["MSIS Density (kg/m^3)", "Latitude (deg)", "Longitude (deg)", "log_Altitude"]
    epsilon = 1e-8
    scaling_file = Path("scaling_params.json")
    if scaling_file.exists():
        with open(scaling_file, "r") as f:
            scaling_params = json.load(f)
    else:
        scaling_params = None
    
    # Load the prediction model.
    MODEL_PATH = Path("density_prediction_patchtst_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTST(seq_len=seq_len, patch_size=2, num_features=len(selected_feature_columns),
                      d_model=128, nhead=4, num_layers=3)
    model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Set propagation parameters.
    dt_integ = 10.0
    total_seconds = 3 * 86400
    steps_integ = int(total_seconds / dt_integ)
    propagation_params = {
        "dt_integ": dt_integ,
        "steps_integ": steps_integ,
        "mass": 1000.0,
        "Cd": 2.2,
        "A": 20.0,
        "sample_rate": 60  # Downsample: 60*10 sec = 600 sec = 10 minutes.
    }
    
    output_dict = {}
    if not args.single_process:
        workers = max(4, os.cpu_count())
        logger.info(f"Using {workers} worker processes.")
        with multiprocessing.Pool(workers) as pool:
            func = partial(process_file, dh=dh, model=model, seq_len=seq_len,
                           selected_feature_columns=selected_feature_columns,
                           epsilon=epsilon, scaling_params=scaling_params,
                           propagation_params=propagation_params)
            results = pool.map(func, file_ids)
            for res in results:
                output_dict.update(res)
    else:
        logger.info("Processing files sequentially.")
        for file_id in file_ids:
            res = process_file(file_id, dh, model, seq_len, selected_feature_columns,
                                 epsilon, scaling_params, propagation_params)
            output_dict.update(res)
            logger.info(f"Processed file {file_id}")
    
    # Save the combined JSON output.
    json_output = json.dumps(output_dict, indent=4)
    output_path = Path("/app/output/prediction.json")
    with open(output_path, "w") as f:
        f.write(json_output)
    logger.info(f"Prediction JSON saved to {output_path}")

if __name__ == "__main__":
    main()
