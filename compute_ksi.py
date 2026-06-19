"""
compute_ksi.py — Dual-timescale Karst Saturation Index for Puerto Rico flash flood events.

KSI_fast(t) = Σ_i P(t-i) · exp(-i·Δt / τ_fast)   [shallow conduit saturation]
KSI_slow(t) = Σ_j P(t-j) · exp(-j·Δt / τ_slow)   [deep aquifer/matrix recharge]
KSI_combined = α·KSI_fast + β·KSI_slow

Usage:
    python compute_ksi.py
    python compute_ksi.py --tau-fast 12 --tau-slow 168 --alpha 0.6 --beta 0.4

Input:  data/processed/flood_rainfall_data.csv
Output: data/processed/flood_events_ksi.csv
        data/processed/ksi_distributions.png

Data limitations (document for ISEF write-up):
  - Rainfall resolution is daily IMERG (not hourly). Δt = 24 h per step.
  - Only 7 days of lookback are available. KSI_slow ideally needs 30 days;
    the current estimate is truncated and will understate aquifer recharge.
  - All 500 events are flood events. Non-flood control events have not yet
    been collected; the discriminability plot uses is_karst as a proxy split
    until controls are added.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TAU_FAST_H: float = 12.0        # hours — shallow/surface saturation
DEFAULT_TAU_SLOW_H: float = 7.0 * 24   # hours — deep karst aquifer recharge
DEFAULT_ALPHA: float = 0.6
DEFAULT_BETA: float = 0.4

HOURS_PER_STEP: float = 24.0   # daily IMERG → each lag step is 24 hours

FAST_LOOKBACK_DAYS: int = 3    # 72 h lookback cap for KSI_fast
SLOW_LOOKBACK_DAYS: int = 7    # 7-day lookback cap for KSI_slow

MISSING_SENTINEL: float = -999.0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
INPUT_CSV = REPO_ROOT / "data" / "processed" / "flood_rainfall_data.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "flood_events_ksi.csv"
OUTPUT_PLOT = REPO_ROOT / "data" / "processed" / "ksi_distributions.png"


# ---------------------------------------------------------------------------
# Core KSI computation
# ---------------------------------------------------------------------------
def compute_ksi(rain_by_lag: list[float], tau_h: float) -> float:
    """
    Exponentially-weighted antecedent precipitation index.

    Parameters
    ----------
    rain_by_lag : list of float
        Daily rainfall values ordered by lag: index 0 = event day (lag 0 h),
        index d = d days prior (lag = d * 24 h). NaN entries are skipped.
    tau_h : float
        Decay timescale in hours. Two physical interpretations:
          tau_fast=12h  → weights drop to ~13% after 1 day (recent storm)
          tau_slow=168h → weights drop to ~87% after 1 day (seasonal baseline)

    Returns
    -------
    float
        KSI at event onset, same units as input rainfall (mm).
    """
    ksi = 0.0
    for d, p in enumerate(rain_by_lag):
        if np.isnan(p):
            continue
        lag_h = d * HOURS_PER_STEP
        ksi += p * np.exp(-lag_h / tau_h)
    return ksi


def ksi_row(
    row: pd.Series,
    tau_fast: float,
    tau_slow: float,
    alpha: float,
    beta: float,
) -> tuple[float, float, float]:
    """Compute (KSI_fast, KSI_slow, KSI_combined) for one event."""
    rain_all = [
        row.get(f"rain_day_minus_{d}", np.nan)
        for d in range(SLOW_LOOKBACK_DAYS)
    ]
    rain_fast = rain_all[:FAST_LOOKBACK_DAYS]

    ksi_fast = compute_ksi(rain_fast, tau_fast)
    ksi_slow = compute_ksi(rain_all, tau_slow)
    ksi_combined = alpha * ksi_fast + beta * ksi_slow
    return ksi_fast, ksi_slow, ksi_combined


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_ksi_distributions(
    df: pd.DataFrame,
    out_path: Path,
    tau_fast: float,
    tau_slow: float,
    alpha: float,
    beta: float,
) -> None:
    """
    Histogram of each KSI component split by is_karst terrain.

    All 500 events are flood events (no non-flood controls yet).
    is_karst (1 = karst terrain, 0 = non-karst) is the primary scientific
    variable for karst quebrada prediction and serves as the binary split.
    """
    karst = df[df["is_karst"] == 1]
    non_karst = df[df["is_karst"] == 0]

    ksi_cols = ["KSI_fast", "KSI_slow", "KSI_combined"]
    subtitles = [
        f"τ={int(tau_fast)}h, α={alpha}\nShallow conduit saturation",
        f"τ={int(tau_slow // 24)}d, β={beta}\nAquifer / matrix recharge",
        f"α · KSI_fast + β · KSI_slow",
    ]
    colors_karst = "#2166ac"
    colors_nonkarst = "#d6604d"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col, subtitle in zip(axes, ksi_cols, subtitles):
        col_min = df[col].min()
        col_max = df[col].max()
        bins = np.linspace(col_min, col_max, 35)

        ax.hist(
            karst[col], bins=bins, alpha=0.65, density=True,
            color=colors_karst, label=f"Karst (n={len(karst)})",
        )
        ax.hist(
            non_karst[col], bins=bins, alpha=0.65, density=True,
            color=colors_nonkarst, label=f"Non-karst (n={len(non_karst)})",
        )

        # Median markers
        ax.axvline(
            karst[col].median(), color=colors_karst,
            linestyle="--", linewidth=1.5, alpha=0.9,
            label=f"Karst median={karst[col].median():.1f}",
        )
        ax.axvline(
            non_karst[col].median(), color=colors_nonkarst,
            linestyle="--", linewidth=1.5, alpha=0.9,
            label=f"Non-karst median={non_karst[col].median():.1f}",
        )

        ax.set_title(f"{col}\n{subtitle}", fontsize=10)
        ax.set_xlabel("mm (rainfall-weighted)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7.5)

    note = (
        "All 500 events are flash floods — no non-flood controls yet. "
        "Split by karst terrain classification (is_karst). "
        f"Daily IMERG (Δt=24 h); fast lookback={FAST_LOOKBACK_DAYS} d, "
        f"slow lookback={SLOW_LOOKBACK_DAYS} d (spec: 30 d)."
    )
    fig.text(0.5, -0.03, note, ha="center", fontsize=8, color="#555555")

    fig.suptitle(
        "KSI Component Distributions — Puerto Rico Flash Flood Events 2015–2024\n"
        "(dashed lines = group medians; histograms normalized to density)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute dual-timescale KSI for flood events.")
    p.add_argument("--tau-fast", type=float, default=DEFAULT_TAU_FAST_H,
                   help="Fast decay timescale in hours (default: 12)")
    p.add_argument("--tau-slow", type=float, default=DEFAULT_TAU_SLOW_H,
                   help="Slow decay timescale in hours (default: 168 = 7 days)")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="Weight on KSI_fast (default: 0.6)")
    p.add_argument("--beta", type=float, default=DEFAULT_BETA,
                   help="Weight on KSI_slow (default: 0.4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tau_fast = args.tau_fast
    tau_slow = args.tau_slow
    alpha = args.alpha
    beta = args.beta

    if not INPUT_CSV.exists():
        sys.exit(f"Input not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} events from {INPUT_CSV.name}")

    # Encode missing-data sentinel as NaN
    rain_cols = [c for c in df.columns if c.startswith("rain_day_minus_")]
    df[rain_cols] = df[rain_cols].replace(MISSING_SENTINEL, np.nan)

    available_days = sorted(int(c.split("_")[-1]) for c in rain_cols)
    print(f"Available lag days in dataset: {available_days}")
    print(
        f"\nParameters:\n"
        f"  τ_fast = {tau_fast} h  (lookback: {FAST_LOOKBACK_DAYS} days)\n"
        f"  τ_slow = {tau_slow} h = {tau_slow/24:.0f} d  (lookback: {SLOW_LOOKBACK_DAYS} days)\n"
        f"  α = {alpha}, β = {beta}"
    )

    # Decay weights for transparency
    print("\nDecay weights applied to each daily lag:")
    print(f"  {'Lag':>5}   {'weight_fast':>12}   {'weight_slow':>12}")
    for d in range(SLOW_LOOKBACK_DAYS):
        wf = np.exp(-d * HOURS_PER_STEP / tau_fast) if d < FAST_LOOKBACK_DAYS else 0.0
        ws = np.exp(-d * HOURS_PER_STEP / tau_slow)
        print(f"  day-{d:>1}:   {wf:>12.4f}   {ws:>12.4f}")

    # Compute KSI for every event
    ksi_results = df.apply(
        lambda row: ksi_row(row, tau_fast, tau_slow, alpha, beta),
        axis=1,
        result_type="expand",
    )
    ksi_results.columns = ["KSI_fast", "KSI_slow", "KSI_combined"]

    df = pd.concat([df, ksi_results], axis=1)

    # Summary statistics
    print("\n── KSI summary statistics ──")
    print(df[["KSI_fast", "KSI_slow", "KSI_combined"]].describe().round(3).to_string())

    # Sanity check: KSI_fast should correlate strongly with same-day rain
    if "rain_day_minus_0" in df.columns:
        print("\n── Pearson r vs rain_day_minus_0 (sanity check) ──")
        for col in ["KSI_fast", "KSI_slow", "KSI_combined"]:
            r = df["rain_day_minus_0"].corr(df[col])
            print(f"  {col}: r = {r:.3f}")

    # Karst vs non-karst median comparison
    if "is_karst" in df.columns:
        print("\n── KSI medians by terrain ──")
        print(df.groupby("is_karst")[["KSI_fast", "KSI_slow", "KSI_combined"]].median().round(2).to_string())

    # Save enriched dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    # Distribution plot
    plot_ksi_distributions(df, OUTPUT_PLOT, tau_fast, tau_slow, alpha, beta)

    print(
        "\n⚠  Limitations to document in ISEF write-up:\n"
        f"  1. Daily IMERG data used (Δt=24 h). KSI_fast (τ={tau_fast}h) is strongly\n"
        f"     dominated by day-0 rainfall (weight=1.00 vs day-1=0.135, day-2=0.018).\n"
        f"     Hourly IMERG would give more discriminating power.\n"
        f"  2. KSI_slow uses {SLOW_LOOKBACK_DAYS}-day lookback vs the target 30 days.\n"
        f"     The slow-recharge signal is truncated; values will be underestimated.\n"
        f"  3. No non-flood control events in the dataset yet. Binary flood/no-flood\n"
        f"     discriminability cannot be computed until controls are added.\n"
        f"     Next step: sample 500 non-flood dates/locations from the same grid."
    )


if __name__ == "__main__":
    main()
