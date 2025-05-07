import logging
import sys
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from modules.utils import (
    ensure_directory_exists
)
# Configure default logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Seaborn style
sns.set(style="whitegrid")

def resource_path(relative_path: str) -> Path:
    """
    Resolve a file path for both normal running and PyInstaller 'onefile' mode.
    If running under PyInstaller, files are unpacked to sys._MEIPASS.
    """
    if getattr(sys, 'frozen', False):  # True if running under PyInstaller
        return Path(sys._MEIPASS) / relative_path
    else:
        return Path(__file__).parent.parent / relative_path

def create_plots_output_directories(base_dir: Path) -> Dict[str, Path]:
    """
    Creates directories for saving different types of plots.
    """
    dirs = {
        'intensity_dir': base_dir / 'intensity_plots',
        'hyetograph_dir': base_dir / 'hyetographs',
        'size_dir': base_dir / 'size_distributions',
        'velocity_dir': base_dir / 'velocity_distributions',
        'heatmap_dir': base_dir / 'heatmaps'  # New directory for heatmaps
    }

    for dir_path in dirs.values():
        ensure_directory_exists(dir_path)

    return dirs


# Utility: compute bin edges from bin centers
def compute_bin_edges(bin_centers: List[float]) -> np.ndarray:
    logger.debug("Computing bin edges for centers: %s", bin_centers)
    bc = np.array(bin_centers)
    edges = np.zeros(len(bc) + 1)
    edges[1:-1] = (bc[:-1] + bc[1:]) / 2
    edges[0] = bc[0] - (bc[1] - bc[0]) / 2
    edges[-1] = bc[-1] + (bc[-1] - bc[-2]) / 2
    logger.debug("Computed edges: %s", edges)
    return edges

# Load diameter-velocity mapping
def load_diam_vel_mapping_csv(mapping_file_path: Union[str, Path]) -> Tuple[List[float], List[float]]:
    logger.debug("Entering load_diam_vel_mapping_csv: %s", mapping_file_path)
    try:
        df = pd.read_csv(mapping_file_path, decimal=',')
        req = ['diameters_mm', 'velocities_m_s']
        missing = [c for c in req if c not in df.columns]
        if missing:
            msg = f"Mapping CSV missing columns: {missing}"
            logger.error(msg)
            raise KeyError(msg)
        diameters = df['diameters_mm'].astype(float).tolist()
        velocities = df['velocities_m_s'].astype(float).tolist()
        logger.debug("Loaded diameters (%d) and velocities (%d)", len(diameters), len(velocities))
        return diameters, velocities
    except Exception:
        logger.exception("Failed loading diameter-velocity mapping from %s", mapping_file_path)
        raise

# Label helper
def _get_label(event_id: Optional[int], start_date: Optional[str], end_date: Optional[str]) -> str:
    try:
        if event_id is not None:
            label = f"Event {event_id}"
        elif start_date and end_date:
            label = f"Date Range {start_date} to {end_date}"
        else:
            label = "All Data"
        logger.debug("Generated label: %s", label)
        return label
    except Exception:
        logger.exception("Error forming label for event_id=%s, start_date=%s, end_date=%s", event_id, start_date, end_date)
        return "Data"

# 1. Precipitation Intensity Plots

# 1. Precipitation Intensity Plots
def plot_precipitation_intensity_separate(
    df_event: pd.DataFrame,
    intervals: List[int],
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None
) -> List[plt.Figure]:
    logger.debug("Entering plot_precipitation_intensity_separate with intervals: %s", intervals)
    figures: List[plt.Figure] = []
    try:
        label = _get_label(event_id, start_date, end_date)
        df = df_event.copy()
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df.set_index('Datetime', inplace=True)
        df = df.dropna(subset=['Intensidad'])
        for interval in intervals:
            try:
                acc = df['Intensidad'].resample(f'{interval}T').sum()
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x=acc.index, y=acc.values, ax=ax)
                ax.set_title(f'Precipitation Intensity - {label} - {interval}min')
                ax.set_xlabel('Time'); ax.set_ylabel('Accumulated Intensity (mm)')
                # Month-day and time format
                locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
                formatter = mdates.DateFormatter('%b %d %H:%M')
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    fp = save_dir / f'precip_intensity_{label.replace(" ","_")}_{interval}min.png'
                    fig.savefig(fp)
                    logger.info("Saved intensity plot to %s", fp)
                figures.append(fig)
                plt.close(fig)
            except Exception:
                logger.exception("Failed to plot interval %d intensity", interval)
        return figures
    except Exception:
        logger.exception("Error in plot_precipitation_intensity_separate")
        return figures

# 2. Hyetograph (10-min intervals)
def plot_hyetograph(
    df_event: pd.DataFrame,
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None
) -> Optional[plt.Figure]:
    logger.debug("Entering plot_hyetograph")
    try:
        label = _get_label(event_id, start_date, end_date)
        interval = 10
        df = df_event.copy()
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df.set_index('Datetime', inplace=True)
        df = df.dropna(subset=['Intensidad'])
        acc = df['Intensidad'].resample(f'{interval}T').sum()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=acc.index, y=acc.values, ax=ax, color='teal')
        ax.set_title(f'Hyetograph - {label} - {interval}min')
        ax.set_xlabel('Time'); ax.set_ylabel('Intensity (mm)')
        # Month-day and time format
        locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
        formatter = mdates.DateFormatter('%b %d %H:%M')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fp = save_dir / f'hyetograph_{label.replace(" ","_")}.png'
            fig.savefig(fp)
            logger.info("Saved hyetograph to %s", fp)
        plt.close(fig)
        return fig
    except Exception:
        logger.exception("Error in plot_hyetograph")
        return None

# 3. Total Accumulated Precipitation
def plot_accumulated_precipitation(
    df_event: pd.DataFrame,
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None
) -> Optional[plt.Figure]:
    logger.debug("Entering plot_accumulated_precipitation")
    try:
        label = _get_label(event_id, start_date, end_date)
        total = df_event['Intensidad'].sum()
        fig, ax = plt.subplots(figsize=(4, 6))
        sns.barplot(x=[label], y=[total], ax=ax, color='violet')
        ax.set_ylabel('Accumulated Intensity (mm)'); ax.set_title(f'Total Precipitation - {label}')
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom')
        plt.tight_layout()
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fp = save_dir / f'accum_precip_{label.replace(" ","_")}.png'
            fig.savefig(fp)
            logger.info("Saved accumulated precipitation to %s", fp)
        plt.close(fig)
        return fig
    except Exception:
        logger.exception("Error in plot_accumulated_precipitation")
        return None
    
# Plot size distribution
def plot_size_distribution(
    combined_matrix: np.ndarray,
    diameters: List[float],
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None
) -> Optional[plt.Figure]:
    logger.debug("Entering plot_size_distribution, matrix shape: %s", combined_matrix.shape)
    try:
        label = _get_label(event_id, start_date, end_date)
        size_dist = combined_matrix.sum(axis=0)
        logger.debug("Size distribution head: %s", size_dist[:5])
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=diameters, y=size_dist, ax=ax)
        ax.set_title(f'Size Distribution - {label}')
        ax.set_xlabel('Diameter (mm)'); ax.set_ylabel('Frequency')
        plt.tight_layout()
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fp = save_dir / f'size_dist_{label.replace(" ","_")}.png'
            fig.savefig(fp)
            logger.info("Saved size distribution to %s", fp)
        plt.close(fig)
        return fig
    except Exception:
        logger.exception("Error in plot_size_distribution")
        return None

# Plot velocity distribution
def plot_velocity_distribution(
    combined_matrix: np.ndarray,
    velocities: List[float],
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None
) -> Optional[plt.Figure]:
    logger.debug("Entering plot_velocity_distribution, matrix shape: %s", combined_matrix.shape)
    try:
        label = _get_label(event_id, start_date, end_date)
        vel_dist = combined_matrix.sum(axis=1)
        logger.debug("Velocity distribution head: %s", vel_dist[:5])
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=velocities, y=vel_dist, ax=ax)
        ax.set_title(f'Velocity Distribution - {label}')
        ax.set_xlabel('Velocity (m/s)'); ax.set_ylabel('Frequency')
        plt.tight_layout()
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fp = save_dir / f'vel_dist_{label.replace(" ","_")}.png'
            fig.savefig(fp)
            logger.info("Saved velocity distribution to %s", fp)
        plt.close(fig)
        return fig
    except Exception:
        logger.exception("Error in plot_velocity_distribution")
        return None

# Plot velocity-diameter heatmap with theoretical curve
def plot_velocity_diameter_heatmap(
    combined_matrix: np.ndarray,
    velocities: List[float],
    diameters: List[float],
    event_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[Path] = None
) -> Optional[plt.Figure]:
    logger.debug("Entering plot_velocity_diameter_heatmap with matrix shape %s", combined_matrix.shape)
    try:
        label = _get_label(event_id, start_date, end_date)
        # Theoretical curve data
        D_teorico = np.array([0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,
                              1.375,1.625,1.875,2.125,2.375,2.75,3.25,3.75,4.25,4.75,
                              5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5])
        V_teorico = np.array([0,0.971,1.961,2.700,3.290,3.781,4.202,4.570,4.897,
                              5.329,5.819,6.240,6.608,6.934,7.365,7.856,8.276,8.644,8.971,
                              9.401,9.892,10.313,10.680,11.007,11.438,11.929,12.349,12.717,13.043,13.406,13.790])
        x_edges = compute_bin_edges(diameters)
        y_edges = compute_bin_edges(velocities)
        fig, ax = plt.subplots(figsize=(8,6))
        mesh = ax.pcolormesh(x_edges, y_edges, combined_matrix, cmap='hot_r')
        fig.colorbar(mesh, ax=ax, label='Count')
        ax.plot(D_teorico, V_teorico, 'k--', label='Theoretical')
        ax.set_title(f'Velocity-Diameter Heatmap - {label}')
        ax.set_xlabel('Diameter (mm)'); ax.set_ylabel('Velocity (m/s)')
        ax.legend()
        plt.tight_layout()
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fp = save_dir / f'heatmap_{label.replace(" ","_")}.png'
            fig.savefig(fp)
            logger.info("Saved heatmap to %s", fp)
        plt.close(fig)
        return fig
    except Exception:
        logger.exception("Error in plot_velocity_diameter_heatmap")
        return None
