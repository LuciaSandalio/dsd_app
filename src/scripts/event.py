# src/scripts/event.py

# --- make 'src' importable when running this file directly ---
import sys
from pathlib import Path
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1]  # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# ------------------------------------------------------------

import logging
import sys
from typing import Optional, Dict, Any

import pandas as pd

from modules.utils import (
    load_config,
    ensure_directory_exists,
    configure_logging,
)
from modules.event_identification import (
    run_event_identification_for_site,
)

# ------------------------------------------------------------------------------
# Per-site runner
# ------------------------------------------------------------------------------

def process_site_events(site: str, cfg: Dict[str, Any]) -> None:
    """Load processed CSV for a site and run event identification."""
    paths = cfg.get("paths", {}) or {}
    processed_dir = Path(paths.get("processed_dir", "data/processed")) / site
    events_root = Path(paths.get("events_dir", "data/processed/events"))
    events_site_dir = events_root / site

    ensure_directory_exists(processed_dir)
    ensure_directory_exists(events_site_dir)

    csv_path = processed_dir / "output_data.csv"
    if not csv_path.exists():
        logging.warning(f"[{site}] Processed CSV not found: {csv_path}. Skipping.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"[{site}] Failed to read {csv_path}: {e}", exc_info=True)
        return

    try:
        result = run_event_identification_for_site(
            df_processed=df,
            cfg=cfg,
            site=site,
            processed_site_dir=processed_dir,
            events_site_dir=events_site_dir,
        )
    except Exception as e:
        logging.error(f"[{site}] Event identification failed: {e}", exc_info=True)
        return

    # Log a tidy summary
    n_events = len(result.get("events", []))
    logging.info(
        f"[{site}] Event stage done. "
        f"events={n_events}, "
        f"annotated={result.get('annotated_csv')}, "
        f"summary={result.get('summary_csv')}, "
        f"manifest={result.get('manifest_json')}"
    )

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

def event_main(config_path: Optional[str] = None) -> None:
    """
    Stage 2 — Event identification for each site listed in config.run.sites.
    Inputs:
      data/processed/<site>/output_data.csv
    Outputs (per site, under paths.events_dir/<site>/):
      - annotated_data.csv
      - event_*.csv
      - combined_event_*.npy (optional)
      - combined_period_*.npy (optional)
      - event_summary.csv
      - manifest.json
    """
    cfg = load_config(config_path)

    # Logging (rotate & level from config)
    logs_dir = Path(cfg.get("paths", {}).get("logs_dir", "logs"))
    ensure_directory_exists(logs_dir)
    configure_logging(
        log_file_path=logs_dir / "event.log",
        max_bytes=int(cfg.get("logging", {}).get("rotation", {}).get("max_mb", 10)) * 1024 * 1024,
        backup_count=int(cfg.get("logging", {}).get("rotation", {}).get("backup_count", 5)),
        log_level=getattr(logging, cfg.get("logging", {}).get("level", "INFO").upper(), logging.INFO),
    )

    logging.info("Starting event_main.")

    # Sites to process
    sites = cfg.get("run", {}).get("sites") or ["bosque_alegre", "villa_dique", "pilar"]

    for site in sites:
        try:
            process_site_events(site, cfg)
        except Exception as e:
            logging.error(f"[{site}] Unhandled exception in site processing: {e}", exc_info=True)

    logging.info("Event identification completed. Log saved to logs/event.log")

if __name__ == "__main__":
    try:
        event_main()
    except Exception:
        logging.exception("Fatal error in event_main.")
        sys.exit(1)
