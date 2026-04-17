#!/usr/bin/env python3
# src/scripts/get_dsdfile.py
# Thin CLI wrapper for Stage 1: Merge raw files into Canonical CSV + Matrices.

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Ensure project root in path
_THIS = Path(__file__).resolve()
PROJ_ROOT = _THIS.parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.modules.config import load_config
from src.modules.processing.merge import merge_site_data
from src.modules.io_utils import ensure_dir

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Ingest Raw Data")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--site", help="Specific site to process (optional)")
    return parser.parse_args()

def find_raw_files(raw_dir: Path, start_date: str = None, end_date: str = None) -> Dict[str, List[Path]]:
    """
    Find raw files and group by site.
    Basic implementation: walk raw_dir, identify site from subdir, filter by filename date.
    """
    site_files = {}
    
    # Parse dates
    s_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    e_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
    
    if start_date: 
        # Adjust end date to be inclusive if needed, or just strict compare
        # Assuming end_date is inclusive YYYY-MM-DD
        if e_date: e_date = e_date.replace(hour=23, minute=59, second=59)

    for root, dirs, files in os.walk(raw_dir):
        root_path = Path(root)
        rel_path = root_path.relative_to(raw_dir)
        
        if str(rel_path) == ".": continue
        
        # Site is generally the first main folder
        parts = rel_path.parts
        if not parts: continue
        site = parts[0]
        
        if site_files.get(site) is None:
            site_files[site] = []
            
        for f in files:
            if not f.endswith(".txt"): continue
            
            # Simple date check in filename (YYYYMMDD)
            # Both formats usually have YYYYMMDD in filename (dsd2...20250820... or dsd1...20250102...)
            try:
                # Extract date heuristic
                # Find 8 digits
                import re
                match = re.search(r"(\d{8})", f)
                if match:
                    dstr = match.group(1)
                    fdate = datetime.strptime(dstr, "%Y%m%d")
                    
                    if s_date and fdate < s_date: continue
                    if e_date and fdate > e_date: continue
                    
                    site_files[site].append(root_path / f)
            except:
                continue
                
    return site_files

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    raw_dir = Path(cfg["paths"]["raw_dir"]).resolve()
    processed_dir = Path(cfg["paths"]["processed_dir"]).resolve()
    
    logging.info(f"Looking for files in {raw_dir}")
    site_map = find_raw_files(raw_dir, args.start_date, args.end_date)
    
    total_files = sum(len(fs) for fs in site_map.values())
    logging.info(f"Found {total_files} files across {len(site_map)} sites.")
    
    if args.site:
        if args.site not in site_map:
            logging.warning(f"Site {args.site} not found in raw data.")
            return
        # Process only specific site
        sites_to_process = {args.site: site_map[args.site]}
    else:
        sites_to_process = site_map
        
    for site, files in sites_to_process.items():
        if not files: continue
        
        logging.info(f"Processing site: {site} ({len(files)} files)")
        
        out_csv = processed_dir / site / "output_data.csv"
        out_matrices = processed_dir / site / "matrices"
        
        result = merge_site_data(files, out_csv, out_matrices, cfg, site)
        
        logging.info(f"Site {site} finished: {result.total_rows} rows, {result.matrices_saved} matrices.")
        logging.info(f"Manifest: {result.manifest_path}")

if __name__ == "__main__":
    main()
