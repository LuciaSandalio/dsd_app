# scripts/visualization_dsd.py
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Make local package importable no matter where we run from
HERE = Path(__file__).resolve().parent
SRC_ROOT = HERE.parent
if str(SRC_ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(SRC_ROOT))

from modules.utils import ensure_directory_exists, load_config  # type: ignore
from modules.visualization import (
    plot_hyetograph,
    plot_precipitation_intensity_separate,
    plot_accumulated_precipitation,
    plot_size_distribution,
    plot_velocity_distribution,
    plot_velocity_diameter_heatmap,
    load_diam_vel_mapping_from_config,
)

# ----------------------------- logging ---------------------------------
LOGS_DIR = SRC_ROOT / "logs"
ensure_directory_exists(LOGS_DIR)
LOG_FILE = LOGS_DIR / "visualization.log"

logger = logging.getLogger("visualization_dsd")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

# ------------------------- path resolution ------------------------------
def _resolve_to_src_root(p: Path) -> Path:
    return p if p.is_absolute() else (SRC_ROOT / p)

def _find_events_root(cfg: dict) -> Path:
    # 1) try config
    processed_dir = Path((cfg.get("io", {}) or {}).get("processed_dir", "data/processed"))
    candidate = _resolve_to_src_root(processed_dir) / "events"
    candidates = [candidate]

    # 2) common fallbacks
    candidates += [
        SRC_ROOT / "data" / "processed" / "events",
        SRC_ROOT.parent / "data" / "processed" / "events",
    ]

    for c in candidates:
        if c.exists():
            return c

    # 3) last resort: search
    found = list(SRC_ROOT.rglob("events"))
    if found:
        return found[0]
    return candidate  # return the main guess so the error message is clear


# ----------------------------- main ------------------------------------
def main() -> None:
    # Config is optional; we only use it to guess paths
    cfg_path = SRC_ROOT / "config" / "config.yaml"
    try:
        cfg = load_config(cfg_path) if cfg_path.exists() else {}
    except Exception as e:
        logger.warning(f"Could not read config at {cfg_path}: {e}")
        cfg = {}

    events_root = _find_events_root(cfg)
    if not events_root.exists():
        logger.error(f"No events directory found at {events_root}")
        return

    logger.info(f"Events root: {events_root}")

    parsivel = (cfg.get("processing", {}) or {}).get("parsivel", {}) or {}
    ms = parsivel.get("matrix_size", [32, 32])
    matrix_size = ms[0] if isinstance(ms, (list, tuple)) else int(ms)

    diameters, velocities = load_diam_vel_mapping_from_config(cfg, expected_matrix_size=matrix_size)

    # Iterate every site folder under events/
    sites = [p for p in events_root.iterdir() if p.is_dir()]
    if not sites:
        logger.error(f"No site folders found under {events_root}")
        return

    for site_dir in sites:
        site = site_dir.name
        logger.info(f"[{site}] Starting plots…")

        # event list: prefer manifest, else glob CSVs
        manifest = site_dir / "manifest.json"
        event_ids: list[int] = []
        if manifest.exists():
            try:
                with open(manifest, "r", encoding="utf-8") as f:
                    man = json.load(f)
                event_ids = [int(e["id"]) for e in man.get("events", [])]
            except Exception as e:
                logger.warning(f"[{site}] Could not parse manifest.json: {e}")

        if not event_ids:
            event_ids = []
            for csv in sorted(site_dir.glob("event_*.csv")):
                try:
                    eid = int(csv.stem.split("_")[1])
                    event_ids.append(eid)
                except Exception:
                    pass

        if not event_ids:
            logger.info(f"[{site}] No events to plot.")
            continue

        # Output root
        plots_root = SRC_ROOT / "plots" / site
        ensure_directory_exists(plots_root)

        for eid in event_ids:
            out_dir = plots_root / f"Event_{eid}"
            hyeto_dir = out_dir / "hyetographs"
            heatmap_dir = out_dir / "heatmaps"
            size_dir = out_dir / "size_distributions"
            vel_dir = out_dir / "velocity_distributions"
            for p in (hyeto_dir, heatmap_dir, size_dir, vel_dir):
                ensure_directory_exists(p)

            # Load event CSV
            ev_csv = site_dir / f"event_{eid}.csv"
            if not ev_csv.exists():
                logger.warning(f"[{site}] Missing CSV for event {eid}: {ev_csv}")
                continue

            try:
                df = pd.read_csv(ev_csv)
            except Exception as e:
                logger.warning(f"[{site}] Failed reading {ev_csv}: {e}")
                continue

            # --- Time-series plots ---
            plot_hyetograph(df, event_id=eid, save_dir=hyeto_dir)
            plot_precipitation_intensity_separate(df, [10, 60], event_id=eid, save_dir=hyeto_dir)
            plot_accumulated_precipitation(df, event_id=eid, save_dir=hyeto_dir)

            # --- Matrix-based plots (if NPY present) ---
            npy = site_dir / f"combined_event_{eid}.npy"
            if npy.exists():
                try:
                    M = np.load(npy)
                    plot_size_distribution(M, diameters or [], event_id=eid, save_dir=size_dir)
                    plot_velocity_distribution(M, velocities or [], event_id=eid, save_dir=vel_dir)
                    plot_velocity_diameter_heatmap(M, velocities or [], diameters or [], event_id=eid, save_dir=heatmap_dir)
                except Exception as e:
                    logger.warning(f"[{site}] Could not load/plot matrix for event {eid}: {e}")
            else:
                logger.info(f"[{site}] No combined matrix for event {eid} (skipping heatmap/distributions).")

        logger.info(f"[{site}] Done. Plots under {plots_root}")

    logger.info("All sites completed.")

if __name__ == "__main__":
    main()
