#!/usr/bin/env python3
# src/scripts/event.py
# Thin CLI wrapper for Stage 2: Event Identification from Canonical CSV.

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List

# Ensure project root in path
_THIS = Path(__file__).resolve()
PROJ_ROOT = _THIS.parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.modules.config import load_config
from src.modules.processing.events import identify_events

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Event Identification")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--site", help="Specific site to process (optional)")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    processed_dir = Path(cfg["paths"]["processed_dir"]).resolve()
    events_dir_root = Path(cfg["paths"]["events_dir"]).resolve()
    
    # Determine sites to run
    # If site arg provided, run that.
    # Else scan processed_dir for sites.
    sites_to_run = []
    
    if args.site:
        sites_to_run.append(args.site)
    else:
        # Scan dir
        if processed_dir.exists():
            for child in processed_dir.iterdir():
                if child.is_dir() and (child / "output_data.csv").exists():
                    sites_to_run.append(child.name)
    
    if not sites_to_run:
        logging.warning(f"No sites found in {processed_dir}")
        return

    logging.info(f"Running Event Identification for: {sites_to_run}")

    for site in sites_to_run:
        input_csv = processed_dir / site / "output_data.csv"
        site_events_dir = events_dir_root / site
        
        logging.info(f"Processing site: {site}")
        result = identify_events(input_csv, site_events_dir, cfg, site)
        
        if result.get("error"):
            logging.error(f"Error for {site}: {result['error']}")
        else:
            logging.info(f"Site {site}: Found {result['events_count']} events.")
            if result['events_count'] > 0:
                logging.info(f"Summary saved: {result['summary_path']}")

if __name__ == "__main__":
    main()
