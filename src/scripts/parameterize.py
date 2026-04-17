#!/usr/bin/env python3
# src/scripts/parameterize.py
# Thin CLI wrapper for Stage 3: DSD Parameterization (N0, Mu, Lambda).

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
from src.modules.processing.file_index import FileIndex
from src.modules.processing.parameterize import parameterize_event
from src.modules.io_utils import ensure_dir

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: DSD Parameterization")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--site", help="Specific site to process (optional)")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # 1. Base Dirs
    raw_dir = Path(cfg["paths"]["raw_dir"]).resolve()
    processed_dir = Path(cfg["paths"]["processed_dir"]).resolve()
    events_dir_root = Path(cfg["paths"]["events_dir"]).resolve()
    
    # 2. Determine Scope
    sites_to_run = []
    if args.site:
        sites_to_run.append(args.site)
    else:
        if processed_dir.exists():
            for child in processed_dir.iterdir():
                if child.is_dir(): sites_to_run.append(child.name)
                
    if not sites_to_run:
        logging.warning("No sites found to parameterize.")
        return
        
    logging.info(f"Parameterizing sites: {sites_to_run}")

    # 3. Process
    for site in sites_to_run:
        site_events_dir = events_dir_root / site
        if not site_events_dir.exists():
            logging.info(f"No events found for {site}. Skipping.")
            continue
            
        # Initialize Index (Raw + Matrices)
        # This is strictly per-site? Yes.
        logging.info(f"[{site}] Building file index (this may take a moment)...")
        file_index = FileIndex(raw_dir / site, processed_dir, site)
        file_index.build_matrix_index()
        
        # Iterate Events
        event_files = sorted(list(site_events_dir.glob("event_*.csv")))
        logging.info(f"[{site}] Found {len(event_files)} events.")
        
        out_param_dir = cfg["parameterize"]["output_dir"]
        ensure_dir(out_param_dir) # Ensure base exists
        site_param_dir = Path(out_param_dir) / site
        ensure_dir(site_param_dir)
        
        count = 0
        for ev_file in event_files:
            if "summary" in ev_file.name: continue
            
            logging.info(f"[{site}] Parameterizing {ev_file.name}")
            
            output_path = site_param_dir / f"param_{ev_file.name}"
            
            # Manifest path (create distinct dir or side-by-side?)
            # Let's put manifests in data/processed/manifests/site
            manifest_dir = PROJ_ROOT / "data" / "processed" / "manifests" / site
            ensure_dir(manifest_dir)
            manifest_path = manifest_dir / f"manifest_{ev_file.stem}.json"
            
            try:
                parameterize_event(ev_file, output_path, manifest_path, file_index)
                # parameterize_event currently returns None (it saves files directly)
                # The previous code expected a return value dict? 
                # My new module code doesn't return the manifest dict, it saves it.
                # So I'll remove the return check.
                count += 1
            except Exception as e:
                logging.error(f"  Failed: {e}")
        
        logging.info(f"[{site}] Finished. Parameterized {count} events.")

if __name__ == "__main__":
    main()
