# scripts/get_dsdfile.py

#!/usr/bin/env python3
# get_dsdfile.py

import sys
from pathlib import Path
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
from datetime import datetime
import pandas as pd  # Added import for pandas in date validation
import numpy as np
from tqdm import tqdm
import yaml
import multiprocessing

# Ensure this is done before importing from modules
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import custom modules after setting up sys.path
from modules.utils import (
    configure_logging,
    load_config, 
    ensure_directory_exists,
    find_files,
    parse_filename_to_date,
    setup_initial_logging
)
from modules.data_processing import (
    save_dataframe,
    save_matrices,
    process_file
)


def get_dsdfile_main(
    start_date: str = None,
    end_date: str = None,
    config_path: str = "config/config.yaml"
):
    """
    Encapsulates the logic to download/process the DSD .txt files,
    using the provided start/end dates and config path.
    """
    # Argument Parser for configuration file path
    parser = argparse.ArgumentParser(description="Process DSD .txt files and generate outputs.")
    parser.add_argument(
        '--config',
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    CONFIG_FILE = Path(args.config).resolve()

    # Setup initial logging to capture early issues
    setup_initial_logging()

    # Load configuration
    try:
        config = load_config(str(CONFIG_FILE))
    except Exception as e:
        logging.critical(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Extract configuration parameters with validation
    get_dsdfile_config = config.get('get_dsdfile', {})
    root_directory = Path(get_dsdfile_config.get('root_directory', '')).resolve()
    output_directory = Path(get_dsdfile_config.get('output_directory', '')).resolve()
    log_file = get_dsdfile_config.get('log_file_path')
    max_workers = get_dsdfile_config.get('max_workers', None)  # If null, set to CPU count

    # Extract date filters
    start_date_str = get_dsdfile_config.get('start_date')
    end_date_str = get_dsdfile_config.get('end_date')

    #  ensure the log‐file directory exists
    ensure_directory_exists(Path(log_file).parent)

    # Now switch to full logging 
    try:
        configure_logging(log_file)
    except Exception as e:
        logging.critical(f"Logging configuration failed: {e}")
        sys.exit(1)

    logging.info("get_dsdfile.py workflow started")

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Check for missing configuration parameters
    missing_params = [
        param for param in ['root_directory', 'output_directory', 'log_file_path']
        if not get_dsdfile_config.get(param)
    ]
    if missing_params:
        logging.critical(f"Missing configuration parameters: {', '.join(missing_params)}")
        sys.exit(1)

    # Validate directories
    try:
        # Ensure output directory exists
        ensure_directory_exists(output_directory)
        logging.info(f"Output directory validated/created: {output_directory}")
    except Exception as e:
        logging.critical(f"Directory validation failed: {e}")
        sys.exit(1)

    
    try:
        # Define exclusion criteria based on configuration or defaults
        exclude_dirs = [Path(output_directory).resolve()]  # Exclude the output directory to prevent processing log files
        exclude_files = [Path(log_file).name]
        txt_files = find_files(
            root_directory=root_directory,
            file_patterns=['*.txt'],
            exclude_dirs=exclude_dirs,
            exclude_files=exclude_files
        )
        if not txt_files:
            logging.warning(f"No .txt files found in directory: {root_directory}")
            print(f"No .txt files found in directory: {root_directory}")
            return

        # Filter files by date range if both start_date and end_date are provided
        if start_date and end_date:
            filtered_files = []
            for f in txt_files:
                file_dt = parse_filename_to_date(f)
                if file_dt and start_date <= file_dt <= end_date:
                    filtered_files.append(f)
                elif not file_dt:
                    logging.warning(f"Could not parse date from filename: {f}. Skipping.")
            txt_files = filtered_files

        if not txt_files:
            logging.warning("No files within the specified date range.")
            print("No files within the specified date range.")
            return

        logging.info(f"Found {len(txt_files)} files to process within the date range.")

        # 7) Group files by location (first folder under root)
        loc_map: dict = {}
        for f in txt_files:
            rel = Path(f).relative_to(root_directory)
            location = rel.parts[0]
            loc_map.setdefault(location, []).append(f)

        # 8) Process each location separately
        for location, files in loc_map.items():
            logging.info(f"Processing location '{location}' with {len(files)} files")
            out_dir_loc = output_directory / location
            ensure_directory_exists(out_dir_loc)

            data_rows: List[List[float]] = []
            matrix_data: List[Tuple[int, np.ndarray]] = []

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(process_file, fp): fp for fp in files}
                for fut in tqdm(as_completed(future_to_file), total=len(future_to_file), desc=f"Files {location}", unit="file"):
                    fp = future_to_file[fut]
                    try:
                        dr, md = fut.result()
                        data_rows.extend(dr)
                        matrix_data.extend(md)
                    except Exception as e:
                        logging.error(f"Error processing file {fp}: {e}", exc_info=True)

            # 9) Save aggregated CSV
            if data_rows:
                csv_path = out_dir_loc / 'output_data.csv'
                save_dataframe(data_rows, str(csv_path))
                logging.info(f"Saved aggregated data for {location} to {csv_path}")
            else:
                logging.warning(f"No data rows extracted for {location}")

            # 10) Save matrices
            if matrix_data:
                save_matrices(matrix_data, str(out_dir_loc))
                logging.info(f"Saved matrices for {location} under {out_dir_loc}/matrices")
            else:
                logging.warning(f"No matrix data extracted for {location}")

            logging.info("get_dsdfile.py workflow completed")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


"""         data_rows: List[List[float]] = []
        matrix_data: List[Tuple[int, np.ndarray]] = []

        # Determine the number of workers
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
            logging.info(f"max_workers not set. Using CPU count: {max_workers}")

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(process_file, file_path): file_path for file_path in txt_files
            }

            # Initialize tqdm progress bar
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing files", unit="file"):
                file_path = future_to_file[future]
                try:
                    file_data_rows, file_matrix_data = future.result()
                    data_rows.extend(file_data_rows)
                    matrix_data.extend(file_matrix_data)
                    logging.info(f"Successfully processed file: {file_path}")
                except Exception as exc:
                    logging.error(f"File {file_path} generated an exception: {exc}", exc_info=True)

        logging.info("All files have been processed.")

        if data_rows:
            # Save aggregated data
            output_csv_path = Path(output_directory) / "output_data.csv"
            save_dataframe(data_rows, str(output_csv_path))
            logging.info(f"Aggregated data saved to {output_csv_path}")
        else:
            logging.warning("No valid data rows were extracted.")

        if matrix_data:
            # Save matrices
            save_matrices(matrix_data, output_directory)
            logging.info(f"Matrices saved to {output_directory}")
        else:
            logging.warning("No valid matrix data was extracted.")

        logging.info("get_dsdfile.py Workflow Completed.")
        print(f"Processing completed. Log saved to {log_file}")

    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

 """
def _cli_entry_point():
    parser = argparse.ArgumentParser(description="Download/process DSD .txt files.")
    parser.add_argument('--config', default="config/config.yaml", help="Path to config file.")
    parser.add_argument('--start-date', help="Start date (YYYY-MM-DD).")
    parser.add_argument('--end-date', help="End date (YYYY-MM-DD).")
    args = parser.parse_args()

    # If not provided on CLI, read from config
 
    config_data = load_config(args.config)
    get_dsdfile_conf = config_data.get("get_dsdfile", {})

    start_date = args.start_date or get_dsdfile_conf.get("start_date")
    end_date = args.end_date or get_dsdfile_conf.get("end_date")


    get_dsdfile_main(args.start_date, args.end_date, args.config)

if __name__ == "__main__":
    _cli_entry_point()
