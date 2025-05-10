#!/usr/bin/env python3
"""
storm_density_model.py – v3.1
=============================

• Uses the *mean* orbit‑mean‑density from each truth file as a **density
  normalizer** (one scalar that goes into both the feature set and the target
  residual).
• Drops the competition‑specific OD‑RMSE metric; early stopping is now based
  on validation RMSE alone.
• Keeps the lightweight `Config` dataclass, robust I/O guards, and attention‑
  pooled bi‑GRU architecture introduced in v3.0.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib.pyplot as plt  # third‑party import after guard
from datahandler import DataHandler                       # local dependency
# ───────────────────────────── configuration ──────────────────────────────


@dataclass
class Config:
    """Central repository for hyper‑parameters and file locations."""
    omni2_folder: Path = Path("./data/omni2")
    initial_state_folder: Path = Path("./data/initial_state")
    sat_density_folder: Path = Path("./data/sat_density")

    # training
    epochs: int = 50
    patience: int = 24
    lr: float = 3e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.3
    hid: int = 384
    layers: int = 3
    batch_size: int = 64

    # model constants
    num_hist_hours: int = 24 * 60          # 60 days hourly
    forecast_steps: int = 3 * 24 * 6       # 432 × 10 min
    log_eps: float = 1e-14

    # reproducibility
    seed: int = 42

    # auto‑filled
    device: str = field(init=False)

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


CFG: Config = Config()      # global instance visible to DataLoader workers
# ──────────────────────────── constants / features ─────────────────────────
BASE_OMNI_COLS = [
    "f10.7_index", "Lyman_alpha", "ap_index_nT", "Kp_index", "Dst_index_nT",
    "AE_index_nT", "SW_Plasma_Speed_km_s", "SW_Proton_Density_N_cm3",
    "Flow_pressure", "BX_nT_GSE_GSM", "BZ_nT_GSM", "Proton_flux_>10_Mev",
]
RESOLUTIONS_H = [1, 3]
LAG_MINUTES   = [60, 120, 180, 240, 360, 720]

STATIC_RAW = [
    "Semi-major Axis (km)", "Eccentricity", "Inclination (deg)",
    "Altitude (km)", "Latitude (deg)", "Longitude (deg)",
]
EARTH_RADIUS = 6378.137  # km  (WGS‑84)

HIST_FEATURES: List[str] = []
for res in RESOLUTIONS_H:
    suf = "" if res == 1 else f"_{res}h"
    HIST_FEATURES += [f"{c}{suf}" for c in BASE_OMNI_COLS]
    for lag in LAG_MINUTES:
        HIST_FEATURES += [f"{c}{suf}_lag{lag}" for c in BASE_OMNI_COLS]

TOTAL_FEATURES = (
    len(HIST_FEATURES)      # OMNI history
    + len(STATIC_RAW)       # orbital / LLA
    + 6                     # cyclical encodings
    + 1                     # density normalizer
)

# ─────────────────────────── helper functions ──────────────────────────────
def _pad_or_trim(a: np.ndarray, length: int, axis: int = 0) -> np.ndarray:
    """Crop or repeat‑pad `a` along `axis` to exactly `length`."""
    if a.shape[axis] == length:
        return a
    if a.shape[axis] > length:
        sl = [slice(None)] * a.ndim; sl[axis] = slice(-length, None)
        return a[tuple(sl)]
    pad_len = length - a.shape[axis]
    pad = np.repeat(a.take([-1], axis=axis), pad_len, axis=axis)
    return np.concatenate([a, pad], axis=axis)


def _build_resolution(df: pd.DataFrame, hrs: int) -> pd.DataFrame:
    if hrs == 1:
        return df
    coarse = df.resample(f"{hrs}h").mean().ffill()
    return coarse.reindex(df.index, method="ffill")


def clean_omni2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace({99999.99: np.nan, 9999999: np.nan, -1: np.nan}).copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df.set_index("Timestamp", inplace=True)

    base = (
        df[BASE_OMNI_COLS]
        .resample("1h").mean()
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    frames: List[pd.DataFrame] = []
    for res in RESOLUTIONS_H:
        suf = "" if res == 1 else f"_{res}h"
        frames.append(_build_resolution(base, res).add_suffix(suf))

    for lag in LAG_MINUTES:
        shift = base.shift(lag // 60, freq=f"{lag}min")
        for res in RESOLUTIONS_H:
            suf = "" if res == 1 else f"_{res}h"
            frames.append(_build_resolution(shift, res).add_suffix(f"{suf}_lag{lag}"))

    full = pd.concat(frames, axis=1).fillna(0.0)
    return full.reindex(columns=HIST_FEATURES)


def _derive_geodetic(init: pd.Series) -> tuple[float, float, float]:
    sma = float(init["Semi-major Axis (km)"])
    ecc = float(init["Eccentricity"])
    inc = math.radians(float(init["Inclination (deg)"]))
    raan = math.radians(float(init.get("RAAN (deg)", 0.0)))
    argp = math.radians(float(init.get("Arg of Perigee (deg)", 0.0)))
    nu   = math.radians(float(init.get("True Anomaly (deg)", 0.0)))

    p = sma * (1 - ecc * ecc)
    r = p / (1 + ecc * math.cos(nu))
    x_p, y_p = r * math.cos(nu), r * math.sin(nu)

    cosO, sinO = math.cos(raan), math.sin(raan)
    cosi, sini = math.cos(inc), math.sin(inc)
    cosw, sinw = math.cos(argp), math.sin(argp)

    R11 = cosO * cosw - sinO * sinw * cosi
    R12 = -cosO * sinw - sinO * cosw * cosi
    R21 = sinO * cosw + cosO * sinw * cosi
    R22 = -sinO * sinw + cosO * cosw * cosi
    R31 = sinw * sini
    R32 = cosw * sini

    x_eci = R11 * x_p + R12 * y_p
    y_eci = R21 * x_p + R22 * y_p
    z_eci = R31 * x_p + R32 * y_p
    r_mag = math.sqrt(x_eci**2 + y_eci**2 + z_eci**2)

    lat = math.degrees(math.asin(z_eci / r_mag))
    lon = math.degrees(math.atan2(y_eci, x_eci)); lon = ((lon + 180) % 360) - 180
    alt = r_mag - EARTH_RADIUS
    return alt, lat, lon


def static_vec(init: pd.Series) -> np.ndarray:
    alt_ok = pd.notna(init["Altitude (km)"]) and 50 <= float(init["Altitude (km)"]) <= 2e5
    lat_ok = pd.notna(init["Latitude (deg)"]) and -90 <= float(init["Latitude (deg)"]) <= 90
    lon_ok = pd.notna(init["Longitude (deg)"]) and -180 <= float(init["Longitude (deg)"]) <= 180

    if not (alt_ok and lat_ok and lon_ok):
        alt, lat, lon = _derive_geodetic(init)
        init["Altitude (km)"], init["Latitude (deg)"], init["Longitude (deg)"] = alt, lat, lon
        logging.info("Filled invalid LLA for %s -> alt=%.1f lat=%.2f lon=%.2f",
                     init.get("File ID", "unknown"), alt, lat, lon)

    raw = init[STATIC_RAW].astype(float).to_numpy()

    lon_rad = math.radians(float(init["Longitude (deg)"]))
    lon_s, lon_c = math.sin(lon_rad), math.cos(lon_rad)

    ts = pd.to_datetime(init["Timestamp"], utc=True)
    doy_rad = 2 * math.pi * ts.day_of_year / 365.25
    sid_rad = 2 * math.pi * (ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400

    return np.concatenate(
        [raw, [lon_s, lon_c, math.sin(doy_rad), math.cos(doy_rad),
               math.sin(sid_rad), math.cos(sid_rad)]]
    )

# ─────────────────────────────── dataset ───────────────────────────────────
class DensityDS(Dataset):
    """Dataset that outputs (history × features, time‑weighted‑log‑residual)."""

    def __init__(
        self,
        fids: List[str],
        dh: DataHandler,
        sx: QuantileTransformer | None = None,
        sy: StandardScaler | None = None,
        fit: bool = False,
    ):
        self.dh = dh
        self.fids = [fid for fid in fids if self._has_full_target(fid)]
        if (lost := len(fids) - len(self.fids)) > 0:
            logging.info("Skipped %d/%d files lacking %d valid density samples",
                         lost, len(fids), CFG.forecast_steps)

        self.sx = sx or QuantileTransformer(output_distribution="normal")
        self.sy = sy or StandardScaler()
        if fit:
            self._fit_scalers()

    # ————— helpers —————
    def _has_full_target(self, fid: str) -> bool:
        try:
            dens = self.dh.read_csv_data(fid, self.dh.sat_density_folder)[
                "Orbit Mean Density (kg/m^3)"
            ].to_numpy(np.float32)
            return (dens < 1.0).sum() >= CFG.forecast_steps
        except Exception as e:
            logging.warning("Skipping %s: %s", fid, e)
            return False

    def _fit_scalers(self) -> None:
        X, Y = [], []
        for fid in tqdm(self.fids, desc="Fitting scalers"):
            x, y = self._xy(fid)
            X.append(x); Y.append(y)
        self.sx.fit(np.vstack(X).reshape(-1, TOTAL_FEATURES))
        self.sy.fit(np.hstack(Y).reshape(-1, 1))

    @staticmethod
    def _density_normalizer(fid: str, dh: DataHandler) -> float:
        """Return mean truth density (<1 kg m⁻³) for file `fid`."""
        try:
            dens = dh.read_csv_data(fid, dh.sat_density_folder)[
                "Orbit Mean Density (kg/m^3)"
            ].to_numpy(np.float32)
            dens = dens[dens < 1.0]
            if dens.size == 0:
                raise ValueError("no valid density < 1 kg/m³")
            return float(dens.mean())
        except Exception as e:
            logging.warning("%s: fallback normalizer – %s", fid, e)
            return 1e-12

    def _xy(self, fid: str) -> tuple[np.ndarray, np.ndarray]:
        omni = clean_omni2(self.dh.read_csv_data(fid, self.dh.omni2_folder))
        hist = _pad_or_trim(omni.to_numpy(np.float32), CFG.num_hist_hours)

        static = static_vec(self.dh.get_initial_state(fid)).astype(np.float32)
        rho0 = self._density_normalizer(fid, self.dh)

        hist = np.hstack([
            hist,
            np.full((CFG.num_hist_hours, 1), rho0, dtype=np.float32),
            np.repeat(static[None, :], CFG.num_hist_hours, 0),
        ])

        dens = self.dh.read_csv_data(fid, self.dh.sat_density_folder)[
            "Orbit Mean Density (kg/m^3)"
        ].to_numpy(np.float32)
        dens = dens[dens < 1.0][:CFG.forecast_steps]
        if dens.size != CFG.forecast_steps:
            raise RuntimeError("Target length mismatch after earlier filtering")

        y = np.log10(np.clip(dens / rho0, CFG.log_eps, None))
        return hist, y

    # ——— Dataset API ———
    def __len__(self): return len(self.fids)

    def __getitem__(self, i: int):
        x, y = self._xy(self.fids[i])
        x = self.sx.transform(x.reshape(-1, TOTAL_FEATURES)).reshape(
            CFG.num_hist_hours, TOTAL_FEATURES
        )
        y = self.sy.transform(y[:, None]).ravel().astype(np.float32)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# ───────────────────────────── model ───────────────────────────────────────
class _AttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__(); self.attn = nn.Linear(d, 1)
    def forward(self, H):                   # (B,T,D)
        a = torch.softmax(self.attn(H), 1)  # (B,T,1)
        return (a * H).sum(1)               # (B,D)

class BiGRUAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            TOTAL_FEATURES, CFG.hid, CFG.layers,
            dropout=CFG.drop_prob, bidirectional=True, batch_first=True
        )
        self.pool = _AttnPool(CFG.hid * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(CFG.hid * 2),
            nn.Linear(CFG.hid * 2, CFG.hid),
            nn.GELU(),
            nn.Dropout(CFG.drop_prob),
            nn.Linear(CFG.hid, CFG.forecast_steps),
        )
    def forward(self, x):
        H, _ = self.gru(x)
        z = self.pool(H)
        return self.head(z)

# ─────────────────────────── loss / metric ────────────────────────────────
class TimeWeightedMSE(nn.Module):
    """Exponential time‑decay MSE (more weight on early forecast steps)."""

    def __init__(self, steps: int | None = None, dt: float = 600.0, eps: float = 1e-5):
        super().__init__()
        steps = steps or CFG.forecast_steps
        gamma = -math.log(eps) / (steps * dt)
        t = torch.arange(steps).float() * dt
        self.register_buffer("w", torch.exp(-gamma * t))

    def forward(self, pred, target):        # both (B,steps)
        return ((pred - target) ** 2 * self.w).mean()

# ───────────────────────────── training utils ─────────────────────────────
def _train_epoch(model, loader, crit, opt, scaler):
    model.train(); tot = 0.0
    ctx = (lambda: torch.amp.autocast("cuda", enabled=True)) \
        if scaler and CFG.device == "cuda" else contextlib.nullcontext

    for xb, yb in loader:
        xb, yb = xb.to(CFG.device), yb.to(CFG.device)
        opt.zero_grad(set_to_none=True)
        with ctx(): loss = crit(model(xb), yb)
        if scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += loss.item() * xb.size(0)

    return math.sqrt(tot / len(loader.dataset))

def _eval_rmse(model, loader, crit):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(CFG.device), yb.to(CFG.device)
            tot += crit(model(xb), yb).item() * xb.size(0)
    return math.sqrt(tot / len(loader.dataset))

def train(ids: List[str], dh: DataHandler, ckpt: Path):
    tr, va = train_test_split(ids, test_size=0.2, random_state=CFG.seed)
    ds_tr = DensityDS(tr, dh, fit=True)
    ds_va = DensityDS(va, dh, sx=ds_tr.sx, sy=ds_tr.sy)

    if len(ds_va) == 0:
        raise RuntimeError("No validation samples after filtering; "
                           "ensure sat_density_folder has ≥432‑row files.")

    dl_tr = DataLoader(ds_tr, CFG.batch_size, True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, CFG.batch_size, num_workers=4, pin_memory=True)

    net  = BiGRUAttn().to(CFG.device)
    crit = TimeWeightedMSE().to(CFG.device)
    opt  = torch.optim.AdamW(net.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4,
                                                       factor=0.5, min_lr=3e-6)
    scaler = torch.amp.GradScaler(enabled=(CFG.device == "cuda"))

    best_rmse, wait = float("inf"), CFG.patience
    for ep in range(1, CFG.epochs + 1):
        rmse_tr = _train_epoch(net, dl_tr, crit, opt, scaler)
        rmse_va = _eval_rmse(net, dl_va, crit)
        logging.info("ep%03d  RMSE(tr)=%.3f  RMSE(val)=%.3f", ep, rmse_tr, rmse_va)
        sched.step(rmse_va)

        if rmse_va < best_rmse - 1e-4:
            best_rmse, wait = rmse_va, CFG.patience
            torch.save({"model": net.state_dict(), "sx": ds_tr.sx, "sy": ds_tr.sy}, ckpt)
        else:
            wait -= 1
            if wait == 0:
                logging.info("Early stopping."); break

    net, _, _ = load_net(ckpt)
    logging.info("Best validation RMSE = %.3f", _eval_rmse(net, dl_va, crit))

# ───────────────────────────── inference helpers ───────────────────────────
def load_net(ckpt: Path):
    chk = torch.load(ckpt, map_location=CFG.device, weights_only=False)
    net = BiGRUAttn().to(CFG.device); net.load_state_dict(chk["model"]); net.eval()
    return net, chk["sx"], chk["sy"]

def predict_one(fid: str, dh: DataHandler, net, sx, sy) -> Dict[str, List]:
    rho0 = DensityDS._density_normalizer(fid, dh)
    omni = clean_omni2(dh.read_csv_data(fid, dh.omni2_folder))
    hist = _pad_or_trim(omni.to_numpy(np.float32), CFG.num_hist_hours)
    sv   = static_vec(dh.get_initial_state(fid)).astype(np.float32)

    hist = np.hstack([
        hist,
        np.full((CFG.num_hist_hours, 1), rho0, dtype=np.float32),
        np.repeat(sv[None, :], CFG.num_hist_hours, 0),
    ])

    x = sx.transform(hist.reshape(-1, TOTAL_FEATURES)).reshape(
        1, CFG.num_hist_hours, TOTAL_FEATURES
    )
    with torch.inference_mode():
        pred_norm = net(torch.from_numpy(x).to(CFG.device)).cpu().numpy().ravel()

    pred_log = sy.inverse_transform(pred_norm[:, None]).ravel()
    dens = rho0 * (10 ** pred_log - CFG.log_eps)

    t0 = pd.to_datetime(dh.get_initial_state(fid)["Timestamp"], utc=True).round("10min")
    ts = pd.date_range(t0, periods=CFG.forecast_steps, freq="10min", tz="UTC")
    return {"Timestamp": [t.isoformat() for t in ts],
            "Orbit Mean Density (kg/m^3)": dens.tolist()}

# ───────────────────────────── plotting utility ────────────────────────────
def plot_file(fid, pred, dh):
    truth = dh.read_csv_data(fid, dh.sat_density_folder).dropna()
    truth.loc[truth["Orbit Mean Density (kg/m^3)"] > 1.0, "Orbit Mean Density (kg/m^3)"] = 0
    plt.figure(figsize=(10, 4)); plt.yscale("log")
    plt.plot(truth["Timestamp"], truth["Orbit Mean Density (kg/m^3)"], label="truth")
    plt.plot(pd.to_datetime(pred["Timestamp"]), pred["Orbit Mean Density (kg/m^3)"], label="pred")
    plt.xlabel("Time [UTC]"); plt.ylabel("ρ [kg m⁻³]"); plt.legend(); plt.tight_layout()
    out = f"compare_{fid}.png"; plt.savefig(out, dpi=150); plt.close()
    logging.info("Plot saved → %s", out)

def write_json(out: Path, chunk: Dict[str, Dict], lock: Path):
    with FileLock(str(lock)):
        data = json.loads(out.read_text()) if out.exists() else {}
        data.update(chunk); out.write_text(json.dumps(data))
# ─────────────────────────────────── CLI ───────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["train", "predict"], default="predict")
    p.add_argument("--local", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=Path("density_net.pt"))
    p.add_argument("--subset_pct", type=float, default=1.0)
    p.add_argument("--epochs", type=int); p.add_argument("--patience", type=int)
    p.add_argument("--plot", metavar="FID")
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.epochs:   CFG.epochs   = args.epochs
    if args.patience: CFG.patience = args.patience

    dh = DataHandler(logger=logging.getLogger(__name__),
                     omni2_folder=CFG.omni2_folder,
                     initial_state_folder=CFG.initial_state_folder,
                     sat_density_folder=CFG.sat_density_folder)
    logging.info("Device: %s", CFG.device)

    if args.mode == "train":
        fids = dh.get_all_file_ids_from_folder(dh.sat_density_folder)
        if args.subset_pct < 1.0:
            fids = random.sample(fids, max(1, int(len(fids) * args.subset_pct)))
        logging.info("Training on %d samples", len(fids))
        train(fids, dh, args.checkpoint)

    else:   # predict
        net, sx, sy = load_net(args.checkpoint)
        if args.plot:
            res = predict_one(args.plot, dh, net, sx, sy); plot_file(args.plot, res, dh); return

        ids = dh.initial_states["File ID"].unique().tolist()
        if args.subset_pct < 1.0:
            ids = random.sample(ids, max(1, int(len(ids) * args.subset_pct)))

        out, lock = Path("prediction.json"), Path("prediction.json.lock")
        for i in range(0, len(ids), args.chunk):
            batch = ids[i:i + args.chunk]; tic = time.time()
            chunk = {fid: predict_one(fid, dh, net, sx, sy) for fid in batch}
            write_json(out, chunk, lock)
            logging.info("Batch %d‑%d done in %.2fs", i+1, i+len(batch), time.time()-tic)
        logging.info("Predictions written → %s", out)

if __name__ == "__main__":
    main()
