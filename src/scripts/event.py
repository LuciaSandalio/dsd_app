#!/usr/bin/env python3
# event.py
# Determines precipitation events

import sys
from pathlib import Path
import argparse
import logging
import sys
import time
import numpy as np


# Ensure this is done before importing from modules
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import custom modules after setting up sys.path
from modules.utils import (
    configure_logging,
    setup_initial_logging,
    load_config,
    ensure_directory_exists
)
from modules.data_processing import (
    load_data
)
from modules.event_identification import (
    remove_duplicates,
    ensure_continuous_timestamps,
    mark_precipitation_activity,
    identify_precipitation_events,
    combine_matrices_for_event,
    save_annotated_data,
    extract_and_save_events
)

def process_location(
    location: str,
    csv_input_base: Path,
    event_cfg: dict,
    combined_csv_base: Path,
    combined_mat_base: Path,
    event_out_base: Path
) -> None:
    """
    Process a single location: load data, identify events, save annotated CSV,
    individual event CSVs, and combined matrices/CSVs.
    """
    loc_dir = csv_input_base / location
    input_csv = loc_dir / event_cfg.get('csv_input_name', 'output_data.csv')
    if not input_csv.exists():
        logging.warning(f"Skipping {location}: {input_csv} not found.")
        return

    logging.info(f"--- Processing location: {location} ---")

    # Prepare per-location directories
    annotated_csv = loc_dir / 'annotated_data.csv'
    event_dir_loc = event_out_base / location
    matrices_dir_loc = loc_dir / 'matrices'

    for d in [event_dir_loc]:
        ensure_directory_exists(d)

    # Step 1: Load data
    df = load_data(
        input_csv,
        required_columns={'Datetime', 'Timestamp', 'Intensidad'},
        dtype_spec={'Timestamp': 'int64', 'Intensidad': 'float64'}
    )

    # Step 2: Remove duplicates
    df, dup_removed = remove_duplicates(df)
    logging.info(f"Removed {dup_removed} duplicates.")

    # Step 3: Ensure continuous timestamps
    df, missing_ts, freq = ensure_continuous_timestamps(df)

    # Step 4: Mark precipitation activity
    df = mark_precipitation_activity(
        df,
        intensidad_threshold=event_cfg.get('intensidad_threshold', 0.0)
    )

    # Step 5: Identify events and combine matrices per-event
    df, event_count = identify_precipitation_events(
        df,
        min_gap_hours=event_cfg.get('min_gap_hours', 2),
        matrix_directory=str(matrices_dir_loc),
        combined_matrix_directory=str(event_dir_loc)
    )
    logging.info(f"Identified {event_count} events for {location}.")

    # Step 6: Save annotated DataFrame
    try:
        save_annotated_data(df, annotated_csv)
       
    except Exception as e:
        logging.error(f"Failed to save annotated CSV for {location}: {e}")

    # Step 7: Extract and save individual events
    extract_and_save_events(df, event_dir_loc, max_workers=event_cfg.get('max_workers'))

    # Step 8: Combine matrices per-event as CSV and NumPy
    unique_events = df['Precip_Event'].dropna().unique()
    for eid in unique_events:
        eid_int = int(eid)
        ts_list = df[df['Precip_Event'] == eid_int]['Timestamp'].tolist()
        csv_out = event_dir_loc / f'combined_event_{eid_int}.csv'
        result = combine_matrices_for_event(
            event_timestamps=ts_list,
            matrix_directory=str(matrices_dir_loc),
            output_csv_dir=str(csv_out)
        )
        if result and 'combined_event_matrix' in result:
            arr = result['combined_event_matrix']
            np_out = event_dir_loc / f'combined_event_{eid_int}.npy'
            try:
                np.save(np_out, arr)
                logging.info(f"Saved NumPy matrix for {location} event {eid_int} to {np_out}")
            except Exception as e:
                logging.error(f"Failed to save NumPy for {location} event {eid_int}: {e}")

    # Step 9: Combine entire date range
    start_date = event_cfg.get('start_date')
    end_date = event_cfg.get('end_date')
    if start_date and end_date:
        csv_out = event_dir_loc / f'combined_matrix_{start_date}_to_{end_date}.csv'
        result = combine_matrices_for_event(
            start_date=start_date,
            end_date=end_date,
            matrix_directory=str(matrices_dir_loc),
            output_csv_dir=str(csv_out)
        )
        if result and 'date_range_matrix' in result:
            arr = result['date_range_matrix']
            np_out = event_dir_loc / f'combined_matrix_{start_date}_to_{end_date}.npy'
            try:
                np.save(np_out, arr)
                logging.info(f"Saved NumPy matrix for {location} date range to {np_out}")
            except Exception as e:
                logging.error(f"Failed to save NumPy for {location} date range: {e}")

    logging.info(f"--- Completed location: {location} ---")


def event_main(
    start_date: str = None,
    end_date: str = None,
    config_path: str = "config/config.yaml"
):
    """
    Loop over each location subfolder under processed data and run the event workflow.
    """
    # Early console logging
    setup_initial_logging()

    # Load config and set up logging
    config = load_config(config_path)
    event_cfg = config.get('event', {})
    log_file = event_cfg.get('log_file', 'logs/event.log')
    ensure_directory_exists(Path(log_file).parent)
    configure_logging(log_file)
    logging.info("Precipitation Event Identification Workflow Started.")

    # Base directories
    csv_input_base = Path(event_cfg.get('csv_input', 'data/processed/output_data.csv')).resolve().parent
    event_out_base = Path(event_cfg.get('event_directory', 'data/processed/events')).resolve()
    

    # Iterate locations
    for loc in csv_input_base.iterdir():
        if not loc.is_dir():
            continue
        process_location(
            location=loc.name,
            csv_input_base=csv_input_base,
            event_cfg=event_cfg,
            combined_csv_base=event_out_base,
            combined_mat_base=event_out_base,
            event_out_base=event_out_base
        )

    logging.info("Precipitation Event Identification Workflow Completed Successfully.")


def _cli_entry_point():
    parser = argparse.ArgumentParser(description="Identify Precipitation Events per location.")
    parser.add_argument('--config', default="config/config.yaml", help="Path to config file.")
    parser.add_argument('--start-date', help="Start date (YYYY-MM-DD).")
    parser.add_argument('--end-date', help="End date (YYYY-MM-DD).")
    args = parser.parse_args()
    event_main(args.start_date, args.end_date, args.config)

if __name__ == "__main__":
    _cli_entry_point()
