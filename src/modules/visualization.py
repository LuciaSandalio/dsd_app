# modules/visualization.py

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Optional: seaborn only for nicer defaults. Safe if absent.
try:
    import seaborn as sns  # noqa: F401
    sns.set(style="whitegrid")
except Exception:
    pass

# ---------------------------------------------------------------------
# Logger (file + console) — created once
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "visualization.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# ---------------------------------------------------------------------
# Back-compat helper expected elsewhere
# ---------------------------------------------------------------------
def resource_path(relative_path: str) -> Path:
    """
    Resolve a file path for both normal running and PyInstaller 'onefile' mode.
    If running under PyInstaller, files are unpacked to sys._MEIPASS.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).resolve().parent.parent / relative_path

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_directory_exists(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def create_plots_output_directories(base_dir: Path) -> Dict[str, Path]:
    """
    Create (if needed) and return the standard subfolders used by plots.
    """
    dirs = {
        "intensity_dir": base_dir / "intensity_plots",
        "hyetograph_dir": base_dir / "hyetographs",
        "size_dir": base_dir / "size_distributions",
        "velocity_dir": base_dir / "velocity_distributions",
        "heatmap_dir": base_dir / "heatmaps",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs



def compute_bin_edges(bin_centers: List[float]) -> np.ndarray:
    logger.debug("Computing bin edges for centers: %s", bin_centers)
    x = bin_centers
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.array([x[0]-0.5, x[0]+0.5])
    mids = (x[1:] + x[:-1]) / 2.0
    first = x[0] - (mids[0] - x[0])
    last  = x[-1] + (x[-1] - mids[-1])
    edges = np.concatenate([[first], mids, [last]])
    return edges

def _label_for(event_id: Optional[int], start_date: Optional[str], end_date: Optional[str]) -> str:
    if event_id is not None and start_date and end_date:
        return f"Event {event_id} ({start_date} – {end_date})"
    if event_id is not None:
        return f"Event {event_id}"
    if start_date and end_date:
        return f"Date Range {start_date} to {end_date}"
    return "All Data"

# Back-compat alias (old code imported this name)
_get_label = _label_for

def load_diam_vel_mapping_from_config(
    cfg: Dict[str, Any],
    expected_matrix_size: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Read Parsivel bin centers from:
      processing.parsivel.bin_centers.{diameters_mm, velocities_m_s}
    Ensures lengths match the expected matrix size (if provided).
    """
    def _coerce(seq):
        if seq is None:
            return None
        return [float(str(x).replace(",", ".")) for x in seq]

    parsivel = (cfg.get("processing", {}) or {}).get("parsivel", {}) or {}
    centers = (parsivel.get("bin_centers", {}) or {})
    diameters = _coerce(centers.get("diameters_mm"))
    velocities = _coerce(centers.get("velocities_m_s"))

    # Enforce lengths for safety
    if expected_matrix_size is not None:
        n = int(expected_matrix_size)
        if diameters is None or len(diameters) != n:
            logger.warning("Config diameters_mm missing or length!=%d. Using index bins.", n)
            diameters = list(np.arange(n, dtype=float))
        if velocities is None or len(velocities) != n:
            logger.warning("Config velocities_m_s missing or length!=%d. Using index bins.", n)
            velocities = list(np.arange(n, dtype=float))

    # Monotonic sanity (avoid weird labels if someone scrambles the list)
    if len(diameters) > 1 and not all(np.diff(diameters) > 0):
        diameters = sorted(diameters)
        logger.warning("Sorted diameters_mm to be strictly increasing.")
    if len(velocities) > 1 and not all(np.diff(velocities) > 0):
        velocities = sorted(velocities)
        logger.warning("Sorted velocities_m_s to be strictly increasing.")

    return diameters, velocities


def _resolve_centers(user_centers: Optional[List[float]], n: int, name: str) -> np.ndarray:
    """
    Ensure we have exactly n bin centers for plotting a vector of length n (or a matrix with n bins).
    If user_centers length != n, fallback to indices [0..n-1] and warn.
    """
    if user_centers is None:
        return np.arange(n, dtype=float)
    arr = np.asarray(user_centers, dtype=float)
    if arr.size != n:
        logger.warning(
            f"{name} centers length {arr.size} != required {n} from matrix; "
            f"falling back to index bins [0..{n-1}]."
        )
        return np.arange(n, dtype=float)
    return arr

# ---------------------------------------------------------------------
# Depth/Hyetograph
# ---------------------------------------------------------------------
def plot_hyetograph(
    df_event: pd.DataFrame,
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None,
    interval_min: int = 10,
) -> Optional[plt.Figure]:
    """
    Plot hyetograph: convert instantaneous rate (mm/h) into depth (mm) and
    aggregate into fixed minute intervals (default 10 min).
    """
    try:
        label = _label_for(event_id, start_date, end_date)
        df = df_event.copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df["Intensidad"] = pd.to_numeric(df["Intensidad"], errors="coerce")
        df = df.dropna(subset=["Datetime", "Intensidad"]).sort_values("Datetime")
        if df.empty:
            logger.info("No rows to plot hyetograph.")
            return None

        df = df.set_index("Datetime")
        dt_s = df.index.to_series().diff().dt.total_seconds()
        median_dt = dt_s[dt_s > 0].median()
        if not pd.notna(median_dt):
            median_dt = 60.0
        dt_s = dt_s.clip(lower=1).fillna(median_dt)

        # per-sample depth (mm)
        depth_mm = df["Intensidad"] * (dt_s / 3600.0)
        acc = depth_mm.resample(f"{int(interval_min)}T", label="right", closed="right", origin="start_day").sum()

        times = acc.index.to_pydatetime()
        x = mdates.date2num(times)
        width = interval_min / 1440.0

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x, acc.values, width=width, align="center")
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
        fig.autofmt_xdate()

        ax.set_title(f"Hyetograph - {label} - {interval_min} min accum")
        ax.set_xlabel("Time")
        ax.set_ylabel("Rain depth (mm)")
        fig.tight_layout()

        if save_dir:
            ensure_directory_exists(save_dir)
            safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
            fp = Path(save_dir) / f"hyetograph_{safe_label}_{interval_min}min.png"
            fig.savefig(fp, dpi=150)
            logger.info(f"Saved hyetograph to {save_dir}")

        plt.close(fig)
        return fig
    except Exception:
        logger.error("Error in plot_hyetograph", exc_info=True)
        return None

def plot_interval_depth(
    df_event: pd.DataFrame,
    interval_min: int,
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Plot depth per interval directly from Intensity as in hyetograph,
    mainly for quick 10/60-minute bars.
    """
    try:
        return plot_hyetograph(
            df_event=df_event,
            event_id=event_id,
            start_date=start_date,
            end_date=end_date,
            save_dir=save_dir,
            interval_min=interval_min,
        )
    except Exception:
        logger.error("Error in plot_interval_depth", exc_info=True)
        return None

# Back-compat: old API expected a function that accepted a list of intervals
def plot_precipitation_intensity_separate(
    df_event: pd.DataFrame,
    intervals: List[int],
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> List[plt.Figure]:
    figs: List[plt.Figure] = []
    for iv in intervals:
        fig = plot_interval_depth(
            df_event=df_event,
            interval_min=int(iv),
            event_id=event_id,
            start_date=start_date,
            end_date=end_date,
            save_dir=save_dir,
        )
        if fig is not None:
            figs.append(fig)
    return figs

def plot_accumulated_depth_bar(
    df_event: pd.DataFrame,
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """Total accumulated depth (mm) over the event/date range."""
    try:
        label = _label_for(event_id, start_date, end_date)
        df = df_event.copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df["Intensidad"] = pd.to_numeric(df["Intensidad"], errors="coerce")
        df = df.dropna(subset=["Datetime", "Intensidad"]).sort_values("Datetime")
        if df.empty:
            logger.info("No rows to plot accumulated depth.")
            return None

        # As in hyetograph: integrate rate into depth
        dt_s = df["Datetime"].diff().dt.total_seconds()
        median_dt = dt_s[dt_s > 0].median()
        if not pd.notna(median_dt):
            median_dt = 60.0
        dt_s = dt_s.clip(lower=1).fillna(median_dt)
        depth_mm = df["Intensidad"] * (dt_s / 3600.0)
        total = float(depth_mm.sum())

        fig, ax = plt.subplots(figsize=(4, 6))
        ax.bar([label], [total])
        ax.set_ylabel("Accumulated depth (mm)")
        ax.set_title(f"Total Precipitation - {label}")
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom")
        fig.tight_layout()

        if save_dir:
            ensure_directory_exists(save_dir)
            safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
            fp = Path(save_dir) / f"accum_depth_{safe_label}.png"
            fig.savefig(fp, dpi=150)
            logger.info(f"Saved accumulated depth to {save_dir}")
        plt.close(fig)
        return fig
    except Exception:
        logger.error("Error in plot_accumulated_depth_bar", exc_info=True)
        return None


plot_accumulated_precipitation = plot_accumulated_depth_bar
# ---------------------------------------------------------------------
# DSD plots (require a combined matrix)
# ---------------------------------------------------------------------
def plot_size_distribution(
    combined_matrix: np.ndarray,
    diameters: Optional[List[float]],
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Size distribution = sum over velocity axis -> length Ncols.
    If `diameters` length != Ncols, fall back to index bins.
    """
    try:
        M = np.asarray(combined_matrix)
        if M.ndim != 2:
            logger.info("Combined matrix is not 2D; skipping size distribution.")
            return None
        n_rows, n_cols = M.shape
        size_dist = M.sum(axis=0)
        xs = _resolve_centers(diameters, n_cols, "diameter")

        label = _label_for(event_id, start_date, end_date)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(xs, size_dist, align="center")
        ax.set_title(f"Size Distribution - {label}")
        ax.set_xlabel("Diameter bin")
        ax.set_ylabel("Count")
        fig.tight_layout()

        if save_dir:
            ensure_directory_exists(save_dir)
            safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
            fp = Path(save_dir) / f"size_dist_{safe_label}.png"
            fig.savefig(fp, dpi=150)
            logger.info(f"Saved size distribution to {save_dir}")
        plt.close(fig)
        return fig
    except Exception:
        logger.error("Error in plot_size_distribution", exc_info=True)
        return None

def plot_velocity_distribution(
    combined_matrix: np.ndarray,
    velocities: Optional[List[float]],
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Velocity distribution = sum over diameter axis -> length Nrows.
    If `velocities` length != Nrows, fall back to index bins.
    """
    try:
        M = np.asarray(combined_matrix)
        if M.ndim != 2:
            logger.info("Combined matrix is not 2D; skipping velocity distribution.")
            return None
        n_rows, n_cols = M.shape
        vel_dist = M.sum(axis=1)
        ys = _resolve_centers(velocities, n_rows, "velocity")

        label = _label_for(event_id, start_date, end_date)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(ys, vel_dist, align="center")
        ax.set_title(f"Velocity Distribution - {label}")
        ax.set_xlabel("Velocity bin")
        ax.set_ylabel("Count")
        fig.tight_layout()

        if save_dir:
            ensure_directory_exists(save_dir)
            safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
            fp = Path(save_dir) / f"vel_dist_{safe_label}.png"
            fig.savefig(fp, dpi=150)
            logger.info(f"Saved velocity distribution to {save_dir}")
        plt.close(fig)
        return fig
    except Exception:
        logger.error("Error in plot_velocity_distribution", exc_info=True)
        return None


def plot_velocity_diameter_heatmap(
    combined_matrix: np.ndarray,
    velocities: List[float],
    diameters: List[float],
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """Heatmap of velocity (rows) vs diameter (cols) with theoretical curve."""
    try:
        # If your helper is named _label_for, use that; otherwise _get_label.
        label = (_label_for if "_label_for" in globals() else _get_label)(event_id, start_date, end_date)

        # Always start with a real array for safety
        M = np.asarray(combined_matrix, dtype=float)
        if M.ndim != 2:
            logger.warning("Combined matrix is not 2D; skipping heatmap.")
            return None

        n_rows, n_cols = M.shape
        len_v, len_d = len(velocities), len(diameters)

        # Make sure matrix orientation matches (rows=velocities, cols=diameters)
        if (n_rows, n_cols) == (len_v, len_d):
            pass  # already aligned
        elif (n_rows, n_cols) == (len_d, len_v):
            logger.debug("Transposing combined_matrix to match (velocities, diameters).")
            M = M.T
            n_rows, n_cols = M.shape
        else:
            # As a defensive fallback, clip to the common shape.
            logger.warning(
                "Shape mismatch: matrix %s vs bins (vel=%d, diam=%d). "
                "Clipping to min compatible shape.",
                M.shape, len_v, len_d
            )
            n_rows_clip = min(n_rows, len_v)
            n_cols_clip = min(n_cols, len_d)
            M = M[:n_rows_clip, :n_cols_clip]
            velocities = velocities[:n_rows_clip]
            diameters = diameters[:n_cols_clip]
            n_rows, n_cols = M.shape

        # Build edges from bin centers
        x_edges = compute_bin_edges(diameters)
        y_edges = compute_bin_edges(velocities)

        fig, ax = plt.subplots(figsize=(8, 6))
        mesh = ax.pcolormesh(x_edges, y_edges, M, cmap="hot_r", shading="auto")
        cbar = fig.colorbar(mesh, ax=ax, label="Count")

        # Theoretical curve overlay (units: mm & m/s)
        D_teorico = np.array(
            [0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,
             1.375,1.625,1.875,2.125,2.375,2.75,3.25,3.75,4.25,4.75,
             5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5]
        )
        V_teorico = np.array(
            [0,0.971,1.961,2.700,3.290,3.781,4.202,4.570,4.897,
             5.329,5.819,6.240,6.608,6.934,7.365,7.856,8.276,8.644,8.971,
             9.401,9.892,10.313,10.680,11.007,11.438,11.929,12.349,12.717,13.043,13.406,13.790]
        )
        ax.plot(D_teorico, V_teorico, "k--", label="Theoretical")

        ax.set_title(f"Velocity–Diameter Heatmap - {label}")
        ax.set_xlabel("Diameter (mm)")
        ax.set_ylabel("Velocity (m/s)")
        ax.legend()
        fig.tight_layout()

        if save_dir:
            ensure_directory_exists(save_dir)
            safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
            fp = Path(save_dir) / f"heatmap_{safe_label}.png"
            fig.savefig(fp, dpi=150)
            logger.info("Saved heatmap to %s", fp)

        plt.close(fig)
        return fig

    except Exception:
        logger.exception("Error in plot_velocity_diameter_heatmap")
        return None
