"""
generate_controls.py — Generate non-flood control events for binary classification.

Strategy: for each of the 500 flood events, sample one random date at the same
lat/lon where no flood was recorded within EXCLUSION_DAYS. Fetch 7-day daily IMERG
rainfall for each control, copy terrain metadata from the matching flood event,
compute KSI features, and combine everything into a single labelled dataset.

Usage:
    export NASA_USER=your_username
    export NASA_PASSWORD='your_password'
    python generate_controls.py [--seed 42] [--exclusion-days 14]

Outputs:
    data/processed/control_events.csv           — 500 control rows with rain + KSI
    data/processed/combined_dataset.csv         — floods + controls, is_flood label
    data/processed/ksi_distributions_binary.png — KSI split by is_flood
"""

import argparse
import os
import random
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import KSI computation from sibling module (no re-implementation needed)
sys.path.insert(0, str(Path(__file__).parent))
from compute_ksi import (
    compute_ksi,
    ksi_row,
    FAST_LOOKBACK_DAYS,
    SLOW_LOOKBACK_DAYS,
    DEFAULT_TAU_FAST_H,
    DEFAULT_TAU_SLOW_H,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    MISSING_SENTINEL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXCLUSION_DAYS: int = 14
SAMPLE_DATE_START = date(2015, 1, 8)   # leave 7-day IMERG lookback room
SAMPLE_DATE_END = date(2024, 12, 31)

FETCH_SLEEP_S: float = 0.35
RETRY_DELAYS_S: tuple = (5, 15, 45)

REPO_ROOT = Path(__file__).parent
FLOOD_CSV = REPO_ROOT / "data" / "processed" / "flood_events_ksi.csv"
CHECKPOINT_CSV = REPO_ROOT / "data" / "processed" / "_controls_checkpoint.csv"
CONTROL_CSV = REPO_ROOT / "data" / "processed" / "control_events.csv"
COMBINED_CSV = REPO_ROOT / "data" / "processed" / "combined_dataset.csv"
PLOT_PATH = REPO_ROOT / "data" / "processed" / "ksi_distributions_binary.png"


# ---------------------------------------------------------------------------
# NASA IMERG fetch
# ---------------------------------------------------------------------------
def _imerg_url(lat: float, lon: float, date_str: str) -> str:
    """Build OPeNDAP ASCII URL for a single IMERG daily pixel."""
    year = date_str[:4]
    month = date_str[4:6]
    lat_idx = round((lat + 89.95) / 0.1)
    lon_idx = round((lon + 179.95) / 0.1)
    fname = f"3B-DAY.MS.MRG.3IMERG.{date_str}-S000000-E235959.V07B.nc4"
    return (
        f"https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDF.07"
        f"/{year}/{month}/{fname}.ascii"
        f"?precipitation[0][{lon_idx}][{lat_idx}]"
    )


def fetch_daily_imerg(
    lat: float,
    lon: float,
    date_str: str,
    session: requests.Session,
) -> Optional[float]:
    """
    Return daily IMERG precipitation (mm/day) for a single point and date.
    date_str must be YYYYMMDD. Returns None on unrecoverable failure.
    """
    url = _imerg_url(lat, lon, date_str)
    for attempt, wait in enumerate([0] + list(RETRY_DELAYS_S)):
        if wait:
            time.sleep(wait)
        try:
            resp = session.get(url, timeout=45)
            if resp.status_code == 200:
                for line in reversed(resp.text.strip().splitlines()):
                    if "," in line:
                        try:
                            return float(line.split(",")[-1].strip())
                        except ValueError:
                            pass
            elif resp.status_code == 404:
                return None  # date genuinely missing from IMERG archive
        except requests.RequestException:
            pass
        if attempt < len(RETRY_DELAYS_S):
            print(f"      retry {attempt + 1}/{len(RETRY_DELAYS_S)} for {date_str}…", flush=True)
    return None


def fetch_rain_window(
    lat: float,
    lon: float,
    anchor_date: date,
    session: requests.Session,
    days: int = SLOW_LOOKBACK_DAYS,
) -> dict:
    """
    Fetch `days` daily IMERG values ending at anchor_date (inclusive).
    Returns dict keyed rain_day_minus_0 … rain_day_minus_{days-1}.
    Missing fetches stored as NaN.
    """
    rain = {}
    for d in range(days):
        target = anchor_date - timedelta(days=d)
        ds = target.strftime("%Y%m%d")
        val = fetch_daily_imerg(lat, lon, ds, session)
        rain[f"rain_day_minus_{d}"] = val if val is not None else np.nan
        time.sleep(FETCH_SLEEP_S)
    return rain


# ---------------------------------------------------------------------------
# Control date sampling
# ---------------------------------------------------------------------------
def build_blackout_calendar(
    flood_df: pd.DataFrame,
    exclusion_days: int,
) -> dict:
    """
    For each unique (lat, lon), build the set of dates within exclusion_days
    of any recorded flood at that location.
    """
    calendar: dict = {}
    for _, row in flood_df.iterrows():
        loc = (row["lat"], row["lon"])
        flood_date = datetime.strptime(row["date"], "%m/%d/%Y").date()
        blackout = calendar.setdefault(loc, set())
        for delta in range(-exclusion_days, exclusion_days + 1):
            blackout.add(flood_date + timedelta(days=delta))
    return calendar


def _all_dates_in_range(start: date, end: date) -> list:
    n = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n)]


def sample_control_date(
    blackout: set,
    already_used: set,
    all_dates: list,
    rng: random.Random,
    max_tries: int = 2000,
) -> Optional[date]:
    """Draw a random date not in blackout and not already used at this location."""
    for _ in range(max_tries):
        candidate = rng.choice(all_dates)
        if candidate not in blackout and candidate not in already_used:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_binary_distributions(df: pd.DataFrame, out_path: Path) -> None:
    """Density histograms of KSI components split by is_flood label."""
    floods = df[df["is_flood"] == 1]
    controls = df[df["is_flood"] == 0]

    ksi_cols = ["KSI_fast", "KSI_slow", "KSI_combined"]
    subtitles = [
        f"τ={int(DEFAULT_TAU_FAST_H)}h, α={DEFAULT_ALPHA}\nShallow saturation",
        f"τ={int(DEFAULT_TAU_SLOW_H // 24)}d, β={DEFAULT_BETA}\nAquifer recharge",
        "α·KSI_fast + β·KSI_slow",
    ]
    color_flood = "#d6604d"
    color_ctrl = "#4dac26"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, subtitle in zip(axes, ksi_cols, subtitles):
        finite = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        bins = np.linspace(finite.min(), finite.max(), 40)

        ax.hist(floods[col].dropna(), bins=bins, alpha=0.65, density=True,
                color=color_flood, label=f"Flood (n={len(floods)})")
        ax.hist(controls[col].dropna(), bins=bins, alpha=0.65, density=True,
                color=color_ctrl, label=f"Control (n={len(controls)})")
        ax.axvline(floods[col].median(), color=color_flood, linestyle="--",
                   linewidth=1.5, label=f"Flood med={floods[col].median():.1f}")
        ax.axvline(controls[col].median(), color=color_ctrl, linestyle="--",
                   linewidth=1.5, label=f"Control med={controls[col].median():.1f}")
        ax.set_title(f"{col}\n{subtitle}", fontsize=10)
        ax.set_xlabel("mm (rainfall-weighted)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7.5)

    fig.suptitle(
        "KSI Distributions: Flood Events vs Non-Flood Controls\n"
        "Puerto Rico 2015–2024  (dashed = medians)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Binary distribution plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate non-flood controls and combine dataset.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclusion-days", type=int, default=EXCLUSION_DAYS,
                   help="Blackout radius around flood dates in days (default: 14)")
    p.add_argument("--nasa-user", default=os.environ.get("NASA_USER", ""),
                   help="NASA Earthdata username (or set NASA_USER env var)")
    p.add_argument("--nasa-password", default=os.environ.get("NASA_PASSWORD", ""),
                   help="NASA Earthdata password (or set NASA_PASSWORD env var)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.nasa_user or not args.nasa_password:
        sys.exit(
            "NASA Earthdata credentials required.\n"
            "  export NASA_USER=juan360\n"
            "  export NASA_PASSWORD='your_password'\n"
            "Or pass --nasa-user / --nasa-password flags."
        )

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load flood events
    # ------------------------------------------------------------------
    if not FLOOD_CSV.exists():
        sys.exit(f"Flood events not found: {FLOOD_CSV}\nRun compute_ksi.py first.")

    flood_df = pd.read_csv(FLOOD_CSV)
    print(f"Loaded {len(flood_df)} flood events from {FLOOD_CSV.name}")

    # Terrain lookup keyed by (lat, lon) — first occurrence wins
    terrain_by_loc: dict = {}
    for _, row in flood_df.iterrows():
        key = (row["lat"], row["lon"])
        if key not in terrain_by_loc:
            terrain_by_loc[key] = {
                "location": row["location"],
                "elevation_m": row["elevation_m"],
                "slope_deg": row["slope_deg"],
                "soil_compname": row["soil_compname"],
                "soil_muname": row["soil_muname"],
                "is_karst": row["is_karst"],
            }

    # ------------------------------------------------------------------
    # 2. Build blackout calendar and sample one control date per flood event
    # ------------------------------------------------------------------
    print(f"\nBuilding blackout calendar (±{args.exclusion_days} days per flood)…")
    blackout_by_loc = build_blackout_calendar(flood_df, args.exclusion_days)
    all_dates = _all_dates_in_range(SAMPLE_DATE_START, SAMPLE_DATE_END)

    # Track per-location used dates to avoid duplicate control rows
    used_dates_by_loc: dict = {}
    controls_meta: list = []
    skipped = 0

    for idx, row in flood_df.iterrows():
        loc = (row["lat"], row["lon"])
        blackout = blackout_by_loc.get(loc, set())
        used = used_dates_by_loc.setdefault(loc, set())

        ctrl_date = sample_control_date(blackout, used, all_dates, rng)
        if ctrl_date is None:
            print(f"  WARNING: no valid date for event {row['event_id']} — skipping")
            skipped += 1
            continue

        used.add(ctrl_date)
        controls_meta.append({
            "ctrl_idx": idx,
            "ctrl_id": f"ctrl_{idx:04d}",
            "lat": loc[0],
            "lon": loc[1],
            "ctrl_date": ctrl_date,
            "flood_event_id": row["event_id"],
        })

    print(f"Sampled {len(controls_meta)} control dates  ({skipped} skipped)")

    # ------------------------------------------------------------------
    # 3. Fetch IMERG for controls with checkpoint / resume
    # ------------------------------------------------------------------
    session = requests.Session()
    session.auth = (args.nasa_user, args.nasa_password)

    already_done: set = set()
    checkpoint_rows: list = []
    if CHECKPOINT_CSV.exists():
        ckpt = pd.read_csv(CHECKPOINT_CSV)
        already_done = set(ckpt["event_id"].tolist())
        checkpoint_rows = ckpt.to_dict("records")
        print(f"Resuming from checkpoint: {len(already_done)} controls already fetched")

    total = len(controls_meta)
    results: list = list(checkpoint_rows)

    for meta in controls_meta:
        ctrl_id = meta["ctrl_id"]
        if ctrl_id in already_done:
            continue

        ctrl_date = meta["ctrl_date"]
        lat, lon = meta["lat"], meta["lon"]
        loc = (lat, lon)

        print(
            f"[{len(results) + 1:>3}/{total}]  {ctrl_id}  {ctrl_date}  "
            f"({lat:.4f}, {lon:.4f})",
            end="  … ",
            flush=True,
        )

        rain = fetch_rain_window(lat, lon, ctrl_date, session)
        day0 = rain.get("rain_day_minus_0", np.nan)
        print(f"day0={day0:.1f} mm" if not np.isnan(day0) else "day0=NaN")

        terrain = terrain_by_loc[loc]
        row = {
            "event_id": ctrl_id,
            "date": ctrl_date.strftime("%m/%d/%Y"),
            "location": terrain["location"],
            "lat": lat,
            "lon": lon,
            "deaths": 0,
            "flood_cause": "None",
            **rain,
            "elevation_m": terrain["elevation_m"],
            "slope_deg": terrain["slope_deg"],
            "soil_compname": terrain["soil_compname"],
            "soil_muname": terrain["soil_muname"],
            "is_karst": terrain["is_karst"],
            "flood_event_id": meta["flood_event_id"],
        }
        results.append(row)

        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False)

    pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False)

    # ------------------------------------------------------------------
    # 4. Compute KSI for controls
    # ------------------------------------------------------------------
    ctrl_df = pd.DataFrame(results)
    rain_cols = [c for c in ctrl_df.columns if c.startswith("rain_day_minus_")]
    ctrl_df[rain_cols] = ctrl_df[rain_cols].replace(MISSING_SENTINEL, np.nan)

    ksi_results = ctrl_df.apply(
        lambda r: ksi_row(r, DEFAULT_TAU_FAST_H, DEFAULT_TAU_SLOW_H,
                          DEFAULT_ALPHA, DEFAULT_BETA),
        axis=1,
        result_type="expand",
    )
    ksi_results.columns = ["KSI_fast", "KSI_slow", "KSI_combined"]
    ctrl_df = pd.concat([ctrl_df, ksi_results], axis=1)

    ctrl_df.to_csv(CONTROL_CSV, index=False)
    print(f"\nControl events saved → {CONTROL_CSV}  ({ctrl_df.shape[0]} rows)")

    # ------------------------------------------------------------------
    # 5. Combine with flood events and add is_flood label
    # ------------------------------------------------------------------
    flood_out = flood_df.copy()
    flood_out["is_flood"] = 1
    flood_out["flood_event_id"] = flood_out["event_id"]

    ctrl_out = ctrl_df.copy()
    ctrl_out["is_flood"] = 0

    # Align columns to flood template
    all_cols = list(flood_out.columns)
    for col in all_cols:
        if col not in ctrl_out.columns:
            ctrl_out[col] = np.nan
    ctrl_out = ctrl_out[all_cols]

    combined = pd.concat([flood_out, ctrl_out], ignore_index=True)
    combined.to_csv(COMBINED_CSV, index=False)
    print(f"Combined dataset saved → {COMBINED_CSV}  ({combined.shape[0]} rows × {combined.shape[1]} cols)")

    # ------------------------------------------------------------------
    # 6. Summary stats
    # ------------------------------------------------------------------
    print("\n── KSI medians by label ──")
    print(
        combined.groupby("is_flood")[["KSI_fast", "KSI_slow", "KSI_combined"]]
        .median()
        .round(2)
        .rename(index={0: "Control (is_flood=0)", 1: "Flood (is_flood=1)"})
        .to_string()
    )

    fetch_failures = ctrl_df[rain_cols].isna().all(axis=1).sum()
    if fetch_failures:
        print(f"\nControls with all rain values missing (full fetch failure): {fetch_failures}")

    # ------------------------------------------------------------------
    # 7. Binary distribution plot
    # ------------------------------------------------------------------
    plot_binary_distributions(combined, PLOT_PATH)

    if CHECKPOINT_CSV.exists():
        CHECKPOINT_CSV.unlink()

    print(
        f"\nDone.\n"
        f"  Flood events:   {(combined['is_flood'] == 1).sum()}\n"
        f"  Control events: {(combined['is_flood'] == 0).sum()}\n"
        f"  Combined CSV:   {COMBINED_CSV}\n"
    )


if __name__ == "__main__":
    main()
