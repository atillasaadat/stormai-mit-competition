#!/usr/bin/env python3
"""storm_density_model.py – v2.4
================================
– multi‑resolution OMNI (1 h & 3 h) with 1/2/3 h lags
– NRLMSISE‑00 baseline so the net learns residuals
– cyclical static features (lon / day‑of‑year / seconds‑in‑day)
– OD‑RMSE skill score in training loop (perfect = 1, baseline = 0)
"""

from __future__ import annotations

import argparse, json, logging, math, random, time
from math import exp, log
from pathlib import Path
from typing import Dict, List

import numpy as np, pandas as pd, torch
from filelock import FileLock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from datahandler import DataHandler     # ← helper supplied by competition baseline

# ───────────────────────────────── constants ────────────────────────────────
NUM_HIST_HOURS = 24 * 60            # 1 440 (60 days hourly history)
FORECAST_STEPS = 3 * 24 * 6         # 432 (3 days @ 10‑min)
LOG_EPS        = 1e-14

# base OMNI columns (add more if desired)
BASE = [
    "f10.7_index", "Lyman_alpha", "ap_index_nT", "Kp_index", "Dst_index_nT",
    "AE_index_nT", "SW_Plasma_Speed_km_s", "SW_Proton_Density_N_cm3",
    "Flow_pressure", "BX_nT_GSE_GSM", "BZ_nT_GSM"
]
RESOLUTIONS_H = [1, 3]              # 1‑h & 3‑h means
LAG_MINUTES   = [60, 120, 180]      # 1 h / 2 h / 3 h lags

# expand to full feature list
HIST_FEATURES = []
for res in RESOLUTIONS_H:
    suf = "" if res == 1 else f"_{res}h"
    HIST_FEATURES += [f"{c}{suf}" for c in BASE]
    for lag in LAG_MINUTES:
        HIST_FEATURES += [f"{c}{suf}_lag{lag}" for c in BASE]

STATIC_RAW     = ["Semi-major Axis (km)", "Eccentricity", "Inclination (deg)",
                  "Altitude (km)", "Latitude (deg)", "Longitude (deg)"]
TOTAL_FEATURES = len(HIST_FEATURES) + len(STATIC_RAW) + 6 + 1   # +6 cyclical, +1 MSIS
# ───────────────────────────────── helpers ──────────────────────────────────

def _pad_or_trim(a: np.ndarray, length: int, axis: int = 0):
    if a.shape[axis] == length: return a
    if a.shape[axis] > length:
        sl = [slice(None)]*a.ndim; sl[axis] = slice(-length, None)
        return a[tuple(sl)]
    pad_len = length - a.shape[axis]
    rep = np.take(a, [-1], axis=axis)
    pad = np.repeat(rep, pad_len, axis=axis)
    return np.concatenate([a, pad], axis=axis)

def _build_resolution(df: pd.DataFrame, hrs: int):
    if hrs == 1: return df
    coarse = df.resample(f"{hrs}h").mean().ffill()
    return coarse.reindex(df.index, method="ffill")

def clean_omni2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace({99999.99: np.nan, 9999999: np.nan, -1: np.nan}).copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df.set_index("Timestamp", inplace=True)
    base = (df[BASE].resample("1h").mean()
                    .interpolate(limit_direction="both").ffill().bfill())

    frames = {}
    for res in RESOLUTIONS_H:
        suf = "" if res == 1 else f"_{res}h"
        frames[res] = _build_resolution(base, res).add_suffix(suf)

    cols = [frames[r] for r in RESOLUTIONS_H]
    for lag in LAG_MINUTES:
        shift = base.shift(lag//60, freq=f"{lag}min")
        for res in RESOLUTIONS_H:
            suf = "" if res == 1 else f"_{res}h"
            cols.append(_build_resolution(shift, res).add_suffix(f"{suf}_lag{lag}"))
    full = pd.concat(cols, axis=1).fillna(0.0)
    return full.reindex(columns=HIST_FEATURES)

def static_vec(init: pd.Series) -> np.ndarray:
    raw = init[STATIC_RAW].astype(float).to_numpy()
    lon_rad = math.radians(float(init["Longitude (deg)"]))
    lon_s, lon_c = math.sin(lon_rad), math.cos(lon_rad)
    ts = pd.to_datetime(init["Timestamp"], utc=True)
    doy_rad = 2*math.pi*ts.day_of_year/365.25
    sid_rad = 2*math.pi*(ts.hour*3600+ts.minute*60+ts.second)/86400.0
    return np.concatenate([raw,
                           [lon_s, lon_c,
                            math.sin(doy_rad), math.cos(doy_rad),
                            math.sin(sid_rad), math.cos(sid_rad)]])

# ─────────────────────────────── dataset ───────────────────────────────────
class DensityDS(Dataset):
    def __init__(self, fids: List[str], dh, sx=None, sy=None, fit=False):
        self.fids, self.dh = fids, dh
        self.sx = sx or QuantileTransformer(output_distribution="normal")
        self.sy = sy or StandardScaler()
        if fit: self._fit()

    # —— scaler fit ——
    def _fit(self):
        X, Y = [], []
        for fid in tqdm(self.fids, desc="scaler_fit"):
            x, y = self._xy(fid)
            X.append(x); Y.append(y)
        self.sx.fit(np.vstack(X).reshape(-1, TOTAL_FEATURES))
        self.sy.fit(np.hstack(Y).reshape(-1,1))

    # —— MSIS baseline for a sample ——
    def _msis_baseline(self, fid: str) -> float:
        try:
            df = self.dh.read_csv_data(fid, self.dh.sat_density_omni_propagated_folder)
            return float(df["Orbit Mean Density (kg/m^3)"].iloc[0])
        except Exception:
            return 1e-12

    # —— build (hist, target) ——
    def _xy(self, fid: str):
        omni = clean_omni2(self.dh.read_csv_data(fid, self.dh.omni2_folder))
        hist = _pad_or_trim(omni.to_numpy(np.float32), NUM_HIST_HOURS)
        stat = static_vec(self.dh.get_initial_state(fid)).astype(np.float32)
        msis = self._msis_baseline(fid)
        hist = np.hstack([hist,
                          np.full((NUM_HIST_HOURS,1), msis, dtype=np.float32),
                          np.repeat(stat[None,:], NUM_HIST_HOURS,0)])

        dens = self.dh.read_csv_data(fid, self.dh.sat_density_folder)["Orbit Mean Density (kg/m^3)"].to_numpy(np.float32)
        valid = dens < 1.0
        dens = dens[valid] if valid.any() else np.full(FORECAST_STEPS, msis)
        dens = _pad_or_trim(dens, FORECAST_STEPS)
        y_log = np.log10(np.clip(dens/msis, LOG_EPS, None))   # residual ratio
        return hist, y_log

    def __len__(self): return len(self.fids)
    def __getitem__(self, i):
        x, y = self._xy(self.fids[i])
        x = self.sx.transform(x.reshape(-1, TOTAL_FEATURES)).reshape(NUM_HIST_HOURS, TOTAL_FEATURES)
        y = self.sy.transform(y.reshape(-1, 1)).ravel().astype(np.float32)   #  ← add astype
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


# ─────────────────────────────── model ─────────────────────────────────────
class BiGRU(nn.Module):
    def __init__(self, inp=TOTAL_FEATURES, hid=256, layers=3):
        super().__init__()
        self.gru = nn.GRU(inp, hid, layers, dropout=0.3,
                          bidirectional=True, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid*2, hid), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(hid, FORECAST_STEPS))
    def forward(self, x):
        _, h = self.gru(x)
        z = torch.cat([h[-2], h[-1]], 1)
        return self.head(z)

# ──────────────────────────── metrics ──────────────────────────────────────
def _rmse(model, loader, crit, opt=None, device="cpu"):
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb); loss = crit(pred, yb)
        if opt is not None:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*xb.size(0); n += xb.size(0)
    return math.sqrt(tot/n)

def _mae_phys(model, loader, ds, device="cpu"):
    mae, n = 0.0, 0
    for xb, yb in loader:
        with torch.no_grad():
            pred = model(xb.to(device)).cpu().numpy(); yb = yb.numpy()
        p_log = ds.sy.inverse_transform(pred)
        t_log = ds.sy.inverse_transform(yb)
        mae += np.abs((10**p_log - LOG_EPS) - (10**t_log - LOG_EPS)).sum(); n += p_log.size
    return mae/n

def _od_rmse(model, loader, ds, device="cpu", eps=1e-5):
    dur_sec = FORECAST_STEPS * 600.0
    gamma   = -log(eps) / dur_sec
    t_sec   = np.arange(FORECAST_STEPS) * 600.0
    w       = np.exp(-gamma * t_sec)

    num, den = 0.0, 0.0
    for xb, yb in loader:
        with torch.no_grad():
            pred = model(xb.to(device)).cpu().numpy()
        p_log = ds.sy.inverse_transform(pred)
        t_log = ds.sy.inverse_transform(yb.numpy())
        msis  = xb[:,0,-1].numpy().reshape(-1,1)

        dens_p = msis * (10**p_log - LOG_EPS)
        dens_t = msis * (10**t_log - LOG_EPS)
        dens_m = msis

        rmse_p = np.sqrt(((dens_p - dens_t)**2).mean(axis=0))
        rmse_m = np.sqrt(((dens_m - dens_t)**2).mean(axis=0))

        num += (w * (1 - rmse_p / rmse_m)).sum()
        den += w.sum()
    return num / den

# ───────────────────────── training loop ───────────────────────────────────
def train(ids: List[str], dh, ckpt: Path, epochs=120, patience=16, device="cpu"):
    tr, va = train_test_split(ids, test_size=0.2, random_state=42)
    ds_tr = DensityDS(tr, dh, fit=True)
    ds_va = DensityDS(va, dh, sx=ds_tr.sx, sy=ds_tr.sy)
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=64)

    net, crit = BiGRU().to(device), nn.MSELoss()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)

    best_skill, wait = -1.0, patience
    for ep in range(1, epochs+1):
        rmse_tr = _rmse(net, dl_tr, crit, opt, device)
        rmse_va = _rmse(net, dl_va, crit, None, device)
        od_va   = _od_rmse(net, dl_va, ds_va, device)
        logging.info(f"ep{ep:02d}  RMSEtr={rmse_tr:.3f}  RMSEva={rmse_va:.3f}  OD‑RMSE={od_va:.4f}")
        if od_va > best_skill + 1e-4:
            best_skill, wait = od_va, patience
            torch.save({"model":net.state_dict(), "sx":ds_tr.sx, "sy":ds_tr.sy}, ckpt)
        else:
            wait -= 1
            if wait == 0:
                logging.info("early‑stop"); break

    net, _, _ = load_net(ckpt, device)
    final_skill = _od_rmse(net, dl_va, ds_va, device)
    logging.info(f"Validation OD‑RMSE skill = {final_skill:.4f}")

# ─────────────────────────── inference ─────────────────────────────────────
def load_net(ckpt: Path, device="cpu"):
    chk = torch.load(ckpt, map_location=device, weights_only=False)
    net = BiGRU(); net.load_state_dict(chk["model"]); net.to(device).eval()
    return net, chk["sx"], chk["sy"]

def predict_one(fid: str, dh, net, sx, sy, device="cpu") -> Dict[str,List]:
    msis = DensityDS._msis_baseline(DensityDS, fid)
    omni = clean_omni2(dh.read_csv_data(fid, dh.omni2_folder))
    hist = _pad_or_trim(omni.to_numpy(np.float32), NUM_HIST_HOURS)
    sv   = static_vec(dh.get_initial_state(fid)).astype(np.float32)
    hist = np.hstack([hist,
                      np.full((NUM_HIST_HOURS,1), msis, dtype=np.float32),
                      np.repeat(sv[None,:], NUM_HIST_HOURS,0)])
    x = sx.transform(hist.reshape(-1, TOTAL_FEATURES)).reshape(1, NUM_HIST_HOURS, TOTAL_FEATURES)
    with torch.no_grad():
        pred_norm = net(torch.from_numpy(x).to(device)).cpu().numpy().ravel()
    pred_log = sy.inverse_transform(pred_norm.reshape(-1,1)).ravel()
    dens = msis * (10**pred_log - LOG_EPS)

    t0 = pd.to_datetime(dh.get_initial_state(fid)["Timestamp"], utc=True).round("10min")
    ts = pd.date_range(t0, periods=FORECAST_STEPS, freq="10min", tz="UTC")
    return {"Timestamp": [t.isoformat() for t in ts],
            "Orbit Mean Density (kg/m^3)": dens.tolist()}

# ───────────────────────────── utility ─────────────────────────────────────
def plot_file(fid, pred, dh):
    truth = dh.read_csv_data(fid, dh.sat_density_folder).dropna()
    truth.loc[truth["Orbit Mean Density (kg/m^3)"] > 1.0,
              "Orbit Mean Density (kg/m^3)"] = 0
    plt.figure(figsize=(10,4)); plt.yscale("log")
    plt.plot(truth["Timestamp"], truth["Orbit Mean Density (kg/m^3)"], label="truth")
    plt.plot(pd.to_datetime(pred["Timestamp"]), pred["Orbit Mean Density (kg/m^3)"], label="pred")
    plt.legend(); plt.xlabel("Time"); plt.ylabel("ρ (kg/m³)")
    plt.show()
    png = f"compare_{fid}.png"; plt.tight_layout(); plt.savefig(png); plt.close()
    logging.info(f"plot → {png}")

def write_json(path: Path, chunk: Dict[str, Dict], lock: Path):
    with FileLock(str(lock)):
        data = json.loads(path.read_text()) if path.exists() else {}
        data.update(chunk); path.write_text(json.dumps(data))

# ─────────────────────────────── main ──────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train","predict"], default="predict")
    p.add_argument("--local", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=Path("density_net.pt"))
    p.add_argument("--subset_pct", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=16)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot_file", type=str)
    p.add_argument("--chunk", type=int, default=50)
    args = p.parse_args()

    if args.local:
        OUT_JSON = "prediction.json"
        DATA = dict(
            omni2_folder=Path("./data/omni2"),
            initial_state_folder=Path("./data/initial_state"),
            sat_density_folder=Path("./data/sat_density"),
            forcasted_omni2_folder=Path("./data/forcasted_omni2"),
            sat_density_omni_forcasted_folder=Path("./data/sat_density_omni_forcasted"),
            sat_density_omni_propagated_folder=Path("./data/sat_density_omni_propagated"))
    else:
        OUT_JSON = "/app/output/prediction.json"
        DATA = dict(
            omni2_folder=Path("/app/data/dataset/test/omni2"),
            initial_state_file=Path("/app/input_data/initial_states.csv"),
            sat_density_folder=None,
            forcasted_omni2_folder=Path("/app/data/dataset/test/forcasted_omni2"),
            sat_density_omni_forcasted_folder=Path("/app/data/dataset/test/sat_density_omni_forcasted"),
            sat_density_omni_propagated_folder=Path("/app/data/dataset/test/sat_density_omni_propagated"))

    dh = DataHandler(logging.getLogger(__name__), **DATA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.mode == "train":
        ids = sorted(dh.get_all_file_ids_from_folder(dh.sat_density_folder))
        if args.subset_pct < 1.0:
            k = max(1, int(len(ids)*args.subset_pct)); ids = random.sample(ids, k)
        logging.info(f"Training on {len(ids)} samples"); train(ids, dh, args.checkpoint,
                                                               epochs=args.epochs, patience=args.patience, device=device)
    else:
        net, sx, sy = load_net(args.checkpoint, device)
        if args.plot:
            if not args.plot_file: raise ValueError("--plot_file required with --plot")
            plot_file(args.plot_file, predict_one(args.plot_file, dh, net, sx, sy, device), dh)
        else:
            ids = dh.initial_states["File ID"].unique().tolist()
            if args.subset_pct < 1.0:
                k = max(1,int(len(ids)*args.subset_pct)); ids = random.sample(ids,k)
            out, lock = Path(OUT_JSON), Path(f"{OUT_JSON}.lock")
            total = math.ceil(len(ids)/args.chunk)
            for b,i in enumerate(range(0,len(ids),args.chunk),1):
                tic = time.time()
                batch = ids[i:i+args.chunk]
                logging.info(f"Batch {b}/{total}: {batch[0]} … {batch[-1]} (n={len(batch)})")
                chunk = {fid: predict_one(fid, dh, net, sx, sy, device) for fid in batch}
                write_json(out, chunk, lock)
                logging.info(f"Batch {b} finished in {time.time()-tic:.2f}s")
            logging.info(f"Predictions written → {out}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
