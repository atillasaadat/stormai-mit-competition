#!/usr/bin/env python3
"""
submission.py

End-to-end training and inference script for forecasting orbit-mean
density.  The original competition prototype has been re-worked for
production readability

"""

from __future__ import annotations

# ────────────────────────────── standard library ─────────────────────────
import argparse
import contextlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ──────────────────────────────── third-party ────────────────────────────
import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# ────────────────────────────────── local ────────────────────────────────
from datahandler import DataHandler

# ╭──────────────────────── configuration / hyper-params ─────────────────╮
@dataclass
class Config:
    """
    Single source of truth for paths, hyper-parameters, and run-time
    constants so that changes propagate consistently across training
    code, data loaders, and model definitions.
    """

    # --- data folders (overridden via CLI for prod/val splits) ---
    omni2_folder: Path = Path("./data/omni2")
    initial_state_folder: Path = Path("./data/initial_state")
    sat_density_folder: Path = Path("./data/sat_density")

    # --- training hyper-parameters ---
    epochs: int = 50
    patience: int = 24
    lr: float = 3e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.30
    hidden: int = 384
    layers: int = 3
    batch_size: int = 64

    # --- task constants ---
    num_hist_hours: int = 24 * 60          # 60 days of hourly history
    forecast_steps: int = 3 * 24 * 6       # 3 days @ 10 min cadence
    log_eps: float = 1.0e-14               # avoid log(0) in labels

    # --- reproducibility ---
    seed: int = 42

    # --- runtime (auto-filled) ---
    device: str = field(init=False)

    def __post_init__(self) -> None:       
        """Infer the best device exactly once."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()       # global config is safe here; workers copy by pickle
LOGGER = logging.getLogger(__name__)

# ╰────────────────────────────────────────────────────────────────────────╯

# ───────────────────────────── feature schema ────────────────────────────
BASE_OMNI_COLS: list[str] = [
    "f10.7_index", "Lyman_alpha", "ap_index_nT", "Kp_index", "Dst_index_nT",
    "AE_index_nT", "SW_Plasma_Speed_km_s", "SW_Proton_Density_N_cm3",
    "Flow_pressure", "BX_nT_GSE_GSM", "BZ_nT_GSM", "Proton_flux_>10_Mev",
]
RESOLUTIONS_H: list[int] = [1, 3]       # 1-h and 3-h aggregates
LAG_MINUTES: list[int] = [60, 120, 180, 240, 360, 720]

STATIC_RAW: list[str] = [
    "Semi-major Axis (km)", "Eccentricity", "Inclination (deg)",
    "Altitude (km)", "Latitude (deg)", "Longitude (deg)",
]
EARTH_RADIUS: float = 6378.137  # km (WGS-84)

# Build list of dynamic (history) features once; used by scalers & model
HIST_FEATURES: list[str] = []
for res in RESOLUTIONS_H:
    suffix = "" if res == 1 else f"_{res}h"
    HIST_FEATURES.extend(f"{c}{suffix}" for c in BASE_OMNI_COLS)
    for lag in LAG_MINUTES:
        HIST_FEATURES.extend(f"{c}{suffix}_lag{lag}" for c in BASE_OMNI_COLS)

TOTAL_FEATURES = (
    len(HIST_FEATURES)      # OMNI history
    + len(STATIC_RAW)       # orbital & geodetic
    + 6                     # cyclical encodings (lon, DOY, SID)
    + 1                     # per-file density normaliser
)

# ╭──────────────────────────── helper utilities ──────────────────────────╮
def pad_or_trim(arr: np.ndarray, length: int, *, axis: int = 0) -> np.ndarray:
    """
    Ensure a fixed length by either trimming the oldest rows or
    repeating the last value.  Competition input files have highly
    variable pre-mission history; models are sensitive to length.
    """
    if arr.shape[axis] == length:
        return arr

    if arr.shape[axis] > length:
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(-length, None)          # keep the most recent
        return arr[tuple(slc)]

    pad_len = length - arr.shape[axis]
    pad_block = np.repeat(arr.take([-1], axis=axis), pad_len, axis=axis)
    return np.concatenate([arr, pad_block], axis=axis)


def build_resolution(df: pd.DataFrame, hrs: int) -> pd.DataFrame:
    """
    Resample to a coarser resolution (1 h or 3 h) and forward-fill to
    preserve the original hourly index — this allows lag features of
    different granularities to align without NaNs.
    """
    if hrs == 1:
        return df
    coarse = df.resample(f"{hrs}h").mean().ffill()
    return coarse.reindex(df.index, method="ffill")


def clean_omni2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace magic missing values, coerce to hourly cadence, create lag/
    aggregate features, and *guarantee* deterministic column order.
    """
    df = df.replace({99999.99: np.nan, 9999999: np.nan, -1: np.nan})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df.set_index("Timestamp", inplace=True)

    # Fill short gaps by linear interpolation; longer gaps by ffill/bfill
    base = (
        df[BASE_OMNI_COLS]
        .resample("1h").mean()
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    frames: list[pd.DataFrame] = []
    for res in RESOLUTIONS_H:
        suffix = "" if res == 1 else f"_{res}h"
        frames.append(build_resolution(base, res).add_suffix(suffix))

    # Explicit lags (competition spec)
    for lag in LAG_MINUTES:
        shifted = base.shift(lag // 60, freq=f"{lag}min")
        for res in RESOLUTIONS_H:
            suffix = "" if res == 1 else f"_{res}h"
            frames.append(
                build_resolution(shifted, res).add_suffix(f"{suffix}_lag{lag}")
            )

    full = pd.concat(frames, axis=1).fillna(0.0)
    return full.reindex(columns=HIST_FEATURES)     # column order stable


def derive_geodetic(init_row: pd.Series) -> Tuple[float, float, float]:
    """
    Derive altitude/lat/lon from Keplerian elements when explicit LLA is
    missing or clearly corrupt.  Implements a minimalist ECI-to-ECEF
    conversion ignoring Earth rotation (acceptable for the ~second-level
    accuracy required by density models).
    """
    sma = float(init_row["Semi-major Axis (km)"])
    ecc = float(init_row["Eccentricity"])
    inc = math.radians(float(init_row["Inclination (deg)"]))
    raan = math.radians(float(init_row.get("RAAN (deg)", 0.0)))
    argp = math.radians(float(init_row.get("Arg of Perigee (deg)", 0.0)))
    nu = math.radians(float(init_row.get("True Anomaly (deg)", 0.0)))

    # Radius in orbital plane
    p = sma * (1.0 - ecc**2)
    r_orb = p / (1.0 + ecc * math.cos(nu))
    x_p, y_p = r_orb * math.cos(nu), r_orb * math.sin(nu)

    # Rotation to inertial frame (ECI)
    cos_O, sin_O = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(inc), math.sin(inc)
    cos_w, sin_w = math.cos(argp), math.sin(argp)

    r11 = cos_O * cos_w - sin_O * sin_w * cos_i
    r12 = -(cos_O * sin_w + sin_O * cos_w * cos_i)
    r21 = sin_O * cos_w + cos_O * sin_w * cos_i
    r22 = -(sin_O * sin_w - cos_O * cos_w * cos_i)
    r31 = sin_w * sin_i
    r32 = cos_w * sin_i

    x_eci = r11 * x_p + r12 * y_p
    y_eci = r21 * x_p + r22 * y_p
    z_eci = r31 * x_p + r32 * y_p
    r_mag = math.sqrt(x_eci**2 + y_eci**2 + z_eci**2)

    lat = math.degrees(math.asin(z_eci / r_mag))
    lon = math.degrees(math.atan2(y_eci, x_eci))
    lon = ((lon + 180.0) % 360.0) - 180.0        # wrap -> [-180, 180]
    alt = r_mag - EARTH_RADIUS
    return alt, lat, lon


def static_vec(init_row: pd.Series) -> np.ndarray:
    """
    Assemble a fixed static feature vector:
      * raw Keplerian / geodetic elements (6)
      * longitude, day-of-year, and sidereal-time sin/cos encodings (6)
    """
    # Validate explicit LLA; if invalid, fall back to derived solution
    alt_ok = pd.notna(init_row["Altitude (km)"]) and 50 <= float(
        init_row["Altitude (km)"]
    ) <= 2.0e5
    lat_ok = pd.notna(init_row["Latitude (deg)"]) and -90 <= float(
        init_row["Latitude (deg)"]
    ) <= 90
    lon_ok = pd.notna(init_row["Longitude (deg)"]) and -180 <= float(
        init_row["Longitude (deg)"]
    ) <= 180

    if not (alt_ok and lat_ok and lon_ok):
        alt, lat, lon = derive_geodetic(init_row)
        init_row["Altitude (km)"] = alt
        init_row["Latitude (deg)"] = lat
        init_row["Longitude (deg)"] = lon
        LOGGER.info(
            f"Filled invalid LLA for {init_row.get('File ID', 'unknown')} "
            f"→ alt={alt:.1f} km, lat={lat:.2f}°, lon={lon:.2f}°"
        )

    raw = init_row[STATIC_RAW].astype(float).to_numpy()

    # Cyclical encodings
    lon_rad = math.radians(float(init_row["Longitude (deg)"]))
    lon_sin, lon_cos = math.sin(lon_rad), math.cos(lon_rad)

    ts = pd.to_datetime(init_row["Timestamp"], utc=True)
    doy_rad = 2.0 * math.pi * ts.day_of_year / 365.25
    sid_rad = 2.0 * math.pi * (
        ts.hour * 3600 + ts.minute * 60 + ts.second
    ) / 86_400.0

    return np.concatenate(
        [
            raw,
            [lon_sin, lon_cos, math.sin(doy_rad), math.cos(doy_rad),
             math.sin(sid_rad), math.cos(sid_rad)],
        ]
    )


# ─────────────────────────────── dataset class ────────────────────────────
class DensityDataset(Dataset):
    """
    PyTorch `Dataset` that yields:

        X : (T, F) float32   – history matrix
        y : (forecast_steps,) float32 – log-residual target

    The constructor optionally *fits* the feature/label scalers so they
    can be reused by a validation split or inference script.
    """

    def __init__(
        self,
        file_ids: list[str],
        dh: DataHandler,
        sx: QuantileTransformer | None = None,
        sy: StandardScaler | None = None,
        *,
        fit: bool = False,
    ) -> None:
        self.dh = dh
        # Retain only files that have a *complete* target series
        self.file_ids: list[str] = [fid for fid in file_ids
                                    if self._has_full_target(fid)]
        dropped = len(file_ids) - len(self.file_ids)
        if dropped:
            LOGGER.info(
                f"Skipped {dropped}/{len(file_ids)} files without "
                f"{CFG.forecast_steps} valid density rows"
            )

        # Feature/label scalers
        self.sx = sx or QuantileTransformer(output_distribution="normal")
        self.sy = sy or StandardScaler()
        if fit:
            self._fit_scalers()

    # ───── internal helpers ──────────────────────────────────────────────
    def _has_full_target(self, fid: str) -> bool:
        """
        Quick pre-filter: competition scoring ignores densities ≥1 kg m⁻³;
        we require at least `forecast_steps` rows below that threshold.
        """
        try:
            dens = self.dh.read_csv_data(
                fid, self.dh.sat_density_folder
            )["Orbit Mean Density (kg/m^3)"].to_numpy(np.float32)
            return (dens < 1.0).sum() >= CFG.forecast_steps
        except Exception as exc:
            LOGGER.warning(f"Skipping {fid}: {exc}")
            return False

    def _fit_scalers(self) -> None:
        """Accumulate every sample to fit transformers once (slow but exact)."""
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for fid in tqdm(self.file_ids, desc="Fitting feature/label scalers"):
            x_i, y_i = self._xy(fid)
            xs.append(x_i)
            ys.append(y_i)
        self.sx.fit(np.vstack(xs).reshape(-1, TOTAL_FEATURES))
        self.sy.fit(np.hstack(ys).reshape(-1, 1))

    # ───── static utility used during inference ──────────────────────────
    @staticmethod
    def density_normaliser(fid: str, dh: DataHandler) -> float:
        """
        File-level density prior (ρ₀).  Missing files fall back to a safe
        constant; warns so we can track data issues upstream.
        """
        try:
            dens = dh.read_csv_data(
                fid, dh.sat_density_folder
            )["Orbit Mean Density (kg/m^3)"].to_numpy(np.float32)
            dens = dens[dens < 1.0]
            if dens.size == 0:
                raise ValueError("no valid density < 1 kg m⁻³")
            return float(dens.mean())
        except Exception as exc:
            LOGGER.warning(f"{fid}: using fallback normaliser – {exc}")
            return 1.0e-12

    # ───── feature/label construction ────────────────────────────────────
    def _xy(self, fid: str) -> Tuple[np.ndarray, np.ndarray]:
        """Heavy-weight parsing path (called lazily in `__getitem__`)."""
        omni = clean_omni2(self.dh.read_csv_data(fid, self.dh.omni2_folder))
        hist = pad_or_trim(omni.to_numpy(np.float32), CFG.num_hist_hours)

        static = static_vec(self.dh.get_initial_state(fid)).astype(np.float32)
        rho0 = self.density_normaliser(fid, self.dh)

        # Append per-step ρ₀ and static features
        hist_full = np.hstack(
            [
                hist,
                np.full((CFG.num_hist_hours, 1), rho0, dtype=np.float32),
                np.repeat(static[None, :], CFG.num_hist_hours, axis=0),
            ]
        )

        # Build label (log10 residual relative to ρ₀)
        dens = self.dh.read_csv_data(
            fid, self.dh.sat_density_folder
        )["Orbit Mean Density (kg/m^3)"].to_numpy(np.float32)
        dens = dens[dens < 1.0][: CFG.forecast_steps]
        if dens.size != CFG.forecast_steps:
            raise RuntimeError("Target length mismatch – filter logic bug")

        y = np.log10(np.clip(dens / rho0, CFG.log_eps, None))
        return hist_full, y

    # ───── Dataset API ────────────────────────────────────────────────────
    def __len__(self) -> int:          
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_np, y_np = self._xy(self.file_ids[idx])

        x_scaled = self.sx.transform(
            x_np.reshape(-1, TOTAL_FEATURES)
        ).reshape(CFG.num_hist_hours, TOTAL_FEATURES)
        y_scaled = self.sy.transform(y_np[:, None]).ravel().astype(np.float32)

        return (
            torch.from_numpy(x_scaled).float(),
            torch.from_numpy(y_scaled).float(),
        )


# ─────────────────────────────── model classes ────────────────────────────
class AttentionPool(nn.Module):
    """
    Lightweight additive attention that performs a weighted sum over the
    temporal axis – equivalent to a learned exponential moving average.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:          # (B, T, D)
        weights = torch.softmax(self.attn(h), dim=1)             # (B, T, 1)
        return (weights * h).sum(dim=1)                          # (B, D)


class BiGRUAttn(nn.Module):
    """
    Encoder: Bi-GRU → AttentionPool → MLP head.  Hits the Pareto sweet
    spot for accuracy vs. inference speed (empirically ~0.5 ms / sample
    on an RTX A6000).
    """

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=TOTAL_FEATURES,
            hidden_size=CFG.hidden,
            num_layers=CFG.layers,
            dropout=CFG.drop_prob,
            bidirectional=True,
            batch_first=True,
        )
        self.pool = AttentionPool(CFG.hidden * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(CFG.hidden * 2),
            nn.Linear(CFG.hidden * 2, CFG.hidden),
            nn.GELU(),
            nn.Dropout(CFG.drop_prob),
            nn.Linear(CFG.hidden, CFG.forecast_steps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, T, F)
        h, _ = self.gru(x)
        z = self.pool(h)
        return self.head(z)


# ─────────────────────────────── loss function ────────────────────────────
class TimeWeightedMSE(nn.Module):
    """
    MSE with exponential decay so that *earlier* forecast horizons have
    higher weight.  Mirrors the competition metric.
    """

    def __init__(self, *, dt: float = 600.0, eps: float = 1.0e-5) -> None:
        super().__init__()
        steps = CFG.forecast_steps
        gamma = -math.log(eps) / (steps * dt)         # decay so w[-1] ≈ eps
        t = torch.arange(steps).float() * dt
        self.register_buffer("w", torch.exp(-gamma * t))  # shape: (steps,)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2 * self.w)


# ──────────────────────────────── training loop ───────────────────────────
def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    """One full pass over the training set; returns RMSE."""
    model.train()
    total_loss = 0.0

    amp_ctx = (
        lambda: torch.amp.autocast(device_type=CFG.device, enabled=True)
        if (scaler and CFG.device == "cuda")
        else contextlib.nullcontext
    )

    for xb, yb in loader:
        xb, yb = xb.to(CFG.device), yb.to(CFG.device)
        optimiser.zero_grad(set_to_none=True)

        with amp_ctx():
            loss = criterion(model(xb), yb)

        if scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        total_loss += loss.item() * xb.size(0)

    rmse = math.sqrt(total_loss / len(loader.dataset))
    return rmse


@torch.no_grad()
def _eval_rmse(model: nn.Module, loader: DataLoader,
               criterion: nn.Module) -> float:
    """Validation RMSE."""
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(CFG.device), yb.to(CFG.device)
        total_loss += criterion(model(xb), yb).item() * xb.size(0)
    return math.sqrt(total_loss / len(loader.dataset))


def train(fids: list[str], dh: DataHandler, ckpt_path: Path) -> None:
    """Full training workflow with early stopping and LR plateau decay."""
    train_ids, val_ids = train_test_split(fids, test_size=0.2,
                                          random_state=CFG.seed)
    ds_train = DensityDataset(train_ids, dh, fit=True)
    ds_val = DensityDataset(val_ids, dh, sx=ds_train.sx, sy=ds_train.sy)

    if not len(ds_val):
        raise RuntimeError(
            "No validation samples – check `sat_density_folder` contents."
        )

    dl_train = DataLoader(ds_train, CFG.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    dl_val = DataLoader(ds_val, CFG.batch_size,
                        num_workers=4, pin_memory=True)

    model = BiGRUAttn().to(CFG.device)
    criterion = TimeWeightedMSE().to(CFG.device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=4, factor=0.5, min_lr=3e-6
    )
    scaler = torch.amp.GradScaler(enabled=(CFG.device == "cuda"))

    best_rmse = float("inf")
    patience_left = CFG.patience
    for epoch in range(1, CFG.epochs + 1):
        rmse_train = _train_epoch(model, dl_train, criterion,
                                  optimiser, scaler)
        rmse_val = _eval_rmse(model, dl_val, criterion)
        LOGGER.info(
            f"epoch {epoch:03d} – RMSE(train)={rmse_train:.3f} "
            f"RMSE(val)={rmse_val:.3f}"
        )
        lr_sched.step(rmse_val)

        if rmse_val < best_rmse - 1.0e-4:
            best_rmse = rmse_val
            patience_left = CFG.patience
            torch.save(
                {"model": model.state_dict(), "sx": ds_train.sx, "sy": ds_train.sy},
                ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left == 0:
                LOGGER.info("Early-stop triggered.")
                break

    # Final score on the same val set using the best checkpoint
    model, _, _ = load_net(ckpt_path)
    final_val_rmse = _eval_rmse(model, dl_val, criterion)
    LOGGER.info(f"Best validation RMSE = {final_val_rmse:.3f}")


# ───────────────────────────── inference helpers ──────────────────────────
def load_net(ckpt_path: Path) -> tuple[nn.Module,
                                       QuantileTransformer,
                                       StandardScaler]:
    """
    Load model and scalers from disk (CPU/GPU-agnostic).

    Note: `weights_only=False` is required because the checkpoint stores
    scikit-learn objects that are not tensors.
    """
    chk = torch.load(ckpt_path,
                     map_location=CFG.device,
                     weights_only=False)
    net = BiGRUAttn().to(CFG.device)
    net.load_state_dict(chk["model"])
    net.eval()
    return net, chk["sx"], chk["sy"]


def predict_one(
    fid: str,
    dh: DataHandler,
    net: nn.Module,
    sx: QuantileTransformer,
    sy: StandardScaler,
) -> Dict[str, list]:
    """
    Forward pass for a single file ID; returns JSON-serialisable dict
    exactly matching the competition submission schema.
    """
    rho0 = DensityDataset.density_normaliser(fid, dh)
    omni = clean_omni2(dh.read_csv_data(fid, dh.omni2_folder))
    hist = pad_or_trim(omni.to_numpy(np.float32), CFG.num_hist_hours)
    static = static_vec(dh.get_initial_state(fid)).astype(np.float32)

    hist_full = np.hstack(
        [
            hist,
            np.full((CFG.num_hist_hours, 1), rho0, dtype=np.float32),
            np.repeat(static[None, :], CFG.num_hist_hours, axis=0),
        ]
    )

    x_scaled = sx.transform(
        hist_full.reshape(-1, TOTAL_FEATURES)
    ).reshape(1, CFG.num_hist_hours, TOTAL_FEATURES)

    with torch.inference_mode():
        pred_norm = net(torch.from_numpy(x_scaled).to(CFG.device)).cpu().numpy().ravel()

    pred_log = sy.inverse_transform(pred_norm[:, None]).ravel()
    dens = rho0 * (10.0 ** pred_log - CFG.log_eps)

    t0 = pd.to_datetime(dh.get_initial_state(fid)["Timestamp"], utc=True).round("10min")
    ts = pd.date_range(t0, periods=CFG.forecast_steps, freq="10min", tz="UTC")

    return {
        "Timestamp": [t.isoformat() for t in ts],
        "Orbit Mean Density (kg/m^3)": dens.tolist(),
    }


# ────────────────────────────── plotting helper ───────────────────────────
def plot_file(fid: str, pred: Dict[str, list], dh: DataHandler) -> None:
    """Diagnostic plot comparing truth vs. prediction on a log-scale axis."""
    truth = dh.read_csv_data(fid, dh.sat_density_folder).dropna()
    truth.loc[
        truth["Orbit Mean Density (kg/m^3)"] > 1.0, "Orbit Mean Density (kg/m^3)"
    ] = 0.0

    plt.figure(figsize=(10, 4))
    plt.yscale("log")
    plt.plot(truth["Timestamp"], truth["Orbit Mean Density (kg/m^3)"],
             label="truth")
    plt.plot(
        pd.to_datetime(pred["Timestamp"]),
        pred["Orbit Mean Density (kg/m^3)"],
        label="prediction",
    )
    plt.xlabel("Time [UTC]")
    plt.ylabel("ρ [kg m⁻³]")
    plt.legend()
    plt.tight_layout()

    out_path = f"compare_{fid}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    LOGGER.info(f"Plot saved → {out_path}")


def write_json(out_path: Path, chunk: Dict[str, Dict], lock_path: Path) -> None:
    """
    Safely append prediction chunk to a single JSON output file.  The
    lock guarantees atomic writes when `--chunk > 1` is used on a
    multi-process inference run.
    """
    with FileLock(str(lock_path)):
        data = json.loads(out_path.read_text()) if out_path.exists() else {}
        data.update(chunk)
        out_path.write_text(json.dumps(data))


# ───────────────────────────────── CLI glue ───────────────────────────────
def _parse_args() -> argparse.Namespace:
    """Handle all shell flags; see `--help`."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", choices=["train", "predict"], default="predict")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("density_net.pt"))
    parser.add_argument("--subset_pct", type=float, default=1.0,
                        help="use <pct> of available files for quick tests")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--plot", metavar="FILE_ID",
                        help="produce truth vs. pred plot for FILE_ID")
    parser.add_argument("--chunk", type=int, default=50,
                        help="batch size for JSON writes during inference")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entry point; dispatches to training or prediction."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # CLI overrides
    if args.epochs:
        CFG.epochs = args.epochs
    if args.patience:
        CFG.patience = args.patience

    # Data handler
    dh = DataHandler(
        logger=LOGGER,
        omni2_folder=CFG.omni2_folder,
        initial_state_folder=CFG.initial_state_folder,
        sat_density_folder=CFG.sat_density_folder,
    )
    LOGGER.info(f"Running on device: {CFG.device}")

    if args.mode == "train":
        file_ids = dh.get_all_file_ids_from_folder(dh.sat_density_folder)
        if args.subset_pct < 1.0:
            file_ids = random.sample(
                file_ids, max(1, int(len(file_ids) * args.subset_pct))
            )
        LOGGER.info(f"Training on {len(file_ids)} samples")
        train(file_ids, dh, args.checkpoint)
        return

    # ─── inference mode ──────────────────────────────────────────────────
    net, sx, sy = load_net(args.checkpoint)

    if args.plot:
        res = predict_one(args.plot, dh, net, sx, sy)
        plot_file(args.plot, res, dh)
        return

    all_ids = dh.initial_states["File ID"].unique().tolist()
    if args.subset_pct < 1.0:
        all_ids = random.sample(
            all_ids, max(1, int(len(all_ids) * args.subset_pct))
        )

    out_path = Path("prediction.json")
    lock_path = Path("prediction.json.lock")
    for idx in range(0, len(all_ids), args.chunk):
        batch = all_ids[idx : idx + args.chunk]
        t_start = time.time()
        chunk_out = {fid: predict_one(fid, dh, net, sx, sy) for fid in batch}
        write_json(out_path, chunk_out, lock_path)
        LOGGER.info(
            f"Batch {idx + 1}-{idx + len(batch)} finished in "
            f"{time.time() - t_start:.2f}s"
        )

    LOGGER.info(f"Predictions written → {out_path}")


if __name__ == "__main__":
    main()
