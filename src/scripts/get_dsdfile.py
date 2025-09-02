#!/usr/bin/env python3
# src/scripts/get_dsdfile.py
# Process Parsivel DSD .txt files into per-location output_data.csv and matrices/

import sys, os, fnmatch
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm

# ───────────── Path setup ─────────────
_THIS = Path(__file__).resolve()
SRC_DIR = _THIS.parents[1]   # /.../dsd_app/src
PROJ_ROOT = _THIS.parents[2] # /.../dsd_app
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))
if str(PROJ_ROOT) not in sys.path: sys.path.insert(0, str(PROJ_ROOT))

# ─────────── Project imports ──────────
from modules.utils import (
    load_config,
    ensure_directory_exists,
    parse_filename_to_date,
    dedup_and_sort_output_csv,
)
from modules.data_processing import (
    save_dataframe,
    save_matrices,
    process_file,
)

# ─────── Console + file logging ───────
def _configure_console_and_file_logging(log_file: Path) -> None:
    ensure_directory_exists(log_file.parent)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root.addHandler(ch); root.addHandler(fh)

# ───────────── Helpers ────────────────
def _find_files_local(
    root_directory: Path,
    file_patterns = ("*.txt",),
    exclude_dirs: Optional[List[Path]] = None,
    exclude_files: Optional[List[str]] = None,
    exclude_file_patterns: Optional[List[str]] = None,
) -> List[str]:
    root_directory = Path(root_directory).resolve()
    ex_dirs  = {Path(p).resolve() for p in (exclude_dirs or [])}
    ex_files = set(exclude_files or [])
    ex_pats  = list(exclude_file_patterns or [])
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        abs_dir = Path(dirpath).resolve()
        if any(abs_dir == ed or str(abs_dir).startswith(str(ed) + os.sep) for ed in ex_dirs):
            dirnames[:] = []
            continue
        for name in filenames:
            if name in ex_files: continue
            if any(fnmatch.fnmatch(name, pat) for pat in ex_pats): continue
            if any(fnmatch.fnmatch(name, pat) for pat in file_patterns):
                out.append(str(Path(dirpath) / name))
    return out

def _group_by_location(file_list: List[str], root_directory: Path) -> Dict[str, List[str]]:
    loc_map: Dict[str, List[str]] = {}
    for f in file_list:
        rel = Path(f).relative_to(root_directory)
        location = rel.parts[0]  # first subdir under raw_dir = site
        loc_map.setdefault(location, []).append(f)
    return loc_map

# ───────────── Core run ───────────────
def get_dsdfile_main(config_path: str = str(PROJ_ROOT / "config" / "config.yaml")) -> int:
    # Load config (your schema)
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"[get_dsdfile] Failed to load config: {e}", file=sys.stderr)
        return 2

    paths = config.get("paths", {}) or {}
    run   = config.get("run",   {}) or {}

    # Resolve directories from your schema
    raw_dir       = Path(paths.get("raw_dir",   PROJ_ROOT / "data/raw")).resolve()
    processed_dir = Path(paths.get("processed_dir", PROJ_ROOT / "data/processed")).resolve()
    logs_dir      = Path(paths.get("logs_dir",  PROJ_ROOT / "logs")).resolve()
    log_file      = logs_dir / "get_dsdfile.log"

    # Logging to console + file
    _configure_console_and_file_logging(log_file)
    logging.info("get_dsdfile.py started")
    logging.debug(f"Using config: {config_path}")

    # Dates (top-level per your file)
    start_date_str = config.get("start_date")
    end_date_str   = config.get("end_date")

    # Optional site filter (if you want to process only listed sites)
    selected_sites = set(run.get("sites", []) or [])
    if selected_sites:
        logging.info(f"Site filter active: {sorted(selected_sites)}")

    # Validate base dirs
    ensure_directory_exists(processed_dir)
    if not raw_dir.exists():
        logging.critical(f"raw_dir not found: {raw_dir}")
        return 3

    # Discover .txt files (unique, exclude junk + processed output folder)
    txt_files = _find_files_local(
        root_directory=raw_dir,
        file_patterns=['*.txt'],
        exclude_dirs=[processed_dir],
        exclude_files=[log_file.name],
        exclude_file_patterns=["*.tmp*", "*.bak*", "*~"],
    )
    txt_files = sorted(set(txt_files))
    logging.info(f"Discovered {len(txt_files)} .txt files under {raw_dir}")

    # Filter by site list (if provided)
    if selected_sites:
        keep = []
        for f in txt_files:
            try:
                rel = Path(f).relative_to(raw_dir)
                site = rel.parts[0]
                if site in selected_sites:
                    keep.append(f)
            except Exception:
                pass
        txt_files = keep
        logging.info(f"{len(txt_files)} files remain after site filtering")

    # Filter by date range (from filename)
    start_date = end_date = None
    if start_date_str and end_date_str:
        from pandas import to_datetime
        try:
            start_date = to_datetime(start_date_str)
            end_date   = to_datetime(end_date_str)
            if start_date > end_date:
                logging.critical("start_date must be <= end_date")
                return 4
        except Exception as e:
            logging.critical(f"Invalid dates in config: {e}")
            return 4

        filtered = []
        for f in txt_files:
            dt = parse_filename_to_date(f)
            if dt and start_date <= dt <= end_date:
                filtered.append(f)
            elif not dt:
                logging.warning(f"Could not parse date from filename (skipping): {f}")
        txt_files = filtered
        logging.info(f"{len(txt_files)} files remain after date filtering [{start_date_str} → {end_date_str}]")

    if not txt_files:
        logging.warning("No files to process after filtering.")
        return 0

    # Process per location (site = first folder under raw_dir)
    loc_map = _group_by_location(txt_files, raw_dir)
    for location, files in loc_map.items():
        out_dir_loc = processed_dir / location
        ensure_directory_exists(out_dir_loc)
        logging.info(f"Processing site '{location}' with {len(files)} files")

        data_rows: List[List[float]] = []
        matrix_data: List[Tuple[int, np.ndarray]] = []

        # Parallel parse
        with ProcessPoolExecutor(max_workers=(config.get("processing", {}).get("parallel", {}).get("max_workers", None))) as ex:
            futs = {ex.submit(process_file, fp): fp for fp in files}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Files {location}", unit="file"):
                fp = futs[fut]
                try:
                    dr, md = fut.result()
                    data_rows.extend(dr)
                    matrix_data.extend(md)
                except Exception as e:
                    logging.error(f"Error processing file {fp}: {e}", exc_info=True)

        # Save aggregated CSV then dedup (atomic)
        if data_rows:
            csv_path = out_dir_loc / "output_data.csv"
            save_dataframe(data_rows, str(csv_path))
            try:
                before, after = dedup_and_sort_output_csv(csv_path)
                logging.info(f"{location}: output_data.csv deduped {before} → {after}")
            except Exception as e:
                logging.error(f"Dedup failed on {csv_path}: {e}", exc_info=True)
            logging.info(f"Saved aggregated data for {location} to {csv_path}")
        else:
            logging.warning(f"No data rows extracted for {location}")

        # Save matrices
        if matrix_data:
            save_matrices(matrix_data, str(out_dir_loc))
            logging.info(f"Saved matrices for {location} under {out_dir_loc}/matrices")
        else:
            logging.warning(f"No matrix data extracted for {location}")

    logging.info("get_dsdfile.py completed successfully")
    return 0

# ────────────── CLI ──────────────
def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Process DSD .txt files (reads dates/paths from your config.yaml).")
    ap.add_argument("--config", default=str(PROJ_ROOT / "config" / "config.yaml"),
                    help="Path to config.yaml")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    print(f"[get_dsdfile] Using config: {args.config}")
    sys.exit(get_dsdfile_main(config_path=args.config))
