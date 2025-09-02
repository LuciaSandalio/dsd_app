#!/usr/bin/env python3
# scripts/master_workflow.py

import sys
from pathlib import Path
import argparse
import logging
from typing import Optional
import pandas as pd
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
SRC_DIR = _THIS.parents[1]          # /.../dsd_app/src
PROJ_ROOT = _THIS.parents[2]        # /.../dsd_app

print(f"Project root: {PROJ_ROOT}")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from modules.utils import cleanup_output, configure_logging, load_config
from scripts.get_dsdfile import get_dsdfile_main
from scripts.event import event_main
# (viz is lazy-imported only if enabled)

# ── Helpers ──────────────────────────────────────────────────────────────────
def _set_dates_in_config(config_path: Path, start_date: str, end_date: str) -> None:
    cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    # flat + nested (compat)
    cfg["start_date"] = start_date
    cfg["end_date"]   = end_date
    cfg.setdefault("get_dsdfile", {})
    cfg["get_dsdfile"]["start_date"] = start_date
    cfg["get_dsdfile"]["end_date"]   = end_date
    cfg.setdefault("event", {})
    cfg["event"]["start_date"] = start_date
    cfg["event"]["end_date"]   = end_date
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logging.info(f"Updated dates in {config_path}: {start_date} → {end_date}")

# ── Orchestrator ─────────────────────────────────────────────────────────────
def run_master_workflow(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config_path: str = str(PROJ_ROOT / "config" / "config.yaml"),
    enable_visualization: bool = False,
) -> None:
    cfg = load_config(Path(config_path).resolve())
    log_file_path = (cfg.get("workflow", {}) or {}).get(
        "log_file", str(PROJ_ROOT / "logs" / "master_workflow.log")
    )
    configure_logging(log_file_path)
    logging.info("Master Workflow Started")

    # If GUI passed dates, validate and write into config. Otherwise trust config.
    if start_date and end_date:
        s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
        if s > e:
            logging.critical("start_date must be <= end_date")
            sys.exit(1)
        _set_dates_in_config(Path(config_path), start_date, end_date)
        cfg = load_config(Path(config_path).resolve())  # reload

    # 0) Cleanup
    cleanup_output(cfg)
    logging.info("Cleanup completed.")

    # 1) Raw -> processed
    get_dsdfile_main(config_path)

    # 2) Events
    event_main(config_path)

    # 3) Visualization (optional)
    if enable_visualization:
        try:
            from scripts.visualization_dsd import main as visualization_cli
            logging.info("Running visualization stage…")
            rc = visualization_cli()
            if rc != 0:
                logging.error(f"Visualization returned {rc}")
        except Exception:
            logging.exception("Visualization failed (skipped).")

    logging.info("Master Workflow Completed Successfully.")
    print("Master workflow completed successfully.")

# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run DSD master workflow (reads dates from config)")
    ap.add_argument("--config", default=str(PROJ_ROOT / "config" / "config.yaml"))
    ap.add_argument("--start-date")  # optional; GUI usually writes to config
    ap.add_argument("--end-date")    # optional; GUI usually writes to config
    ap.add_argument("--enable-visualization", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_master_workflow(
        start_date=args.start_date,
        end_date=args.end_date,
        config_path=args.config,
        enable_visualization=args.enable_visualization,
    )
