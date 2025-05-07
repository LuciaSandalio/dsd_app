# utils.py

"""
utils.py

This module provides utility functions for the dsd_app, including:
  - Directory management
  - Logging configuration
  - File discovery
  - Data loading for matrices and aggregated data
  - Filename date parsing
"""

import sys
import shutil
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import yaml
from typing import List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import fnmatch
from datetime import datetime


def get_base_path() -> Path:
    """
    Returns the base path of the 'dsd_app' folder.
    
    If running from source, it's the parent of this file's directory
    (which should be 'dsd_app/modules/utils.py' -> parent -> 'modules' -> parent -> 'dsd_app').
    
    If running under PyInstaller with --onefile, files are unpacked to a temp folder 
    accessible via sys._MEIPASS, so we return that path.
    """
    if getattr(sys, 'frozen', False):  # PyInstaller sets this flag
        return Path(sys._MEIPASS)  # type: ignore
    else:
        # __file__ is .../dsd_app/modules/utils.py; go two levels up to get 'dsd_app'
        return Path(__file__).resolve().parent.parent

def setup_initial_logging():
    """
    Sets up basic logging to capture directory creation issues before detailed logging is configured.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def configure_logging(
    log_file_path: Union[str, Path],
    max_bytes: int = 5*1024*1024,
    backup_count: int = 3,
    log_level: int = logging.INFO
) -> None:
    """
    Configures the logging settings with rotation and stream output.

    Parameters:
    - log_file_path (str or Path): Path to the log file.
    - max_bytes (int): Maximum size in bytes before rotating the log file.
    - backup_count (int): Number of backup log files to keep.
    - log_level (int): Logging level (e.g., logging.INFO).

    Raises:
    - Exception: If logging configuration fails.
    """
    try:
        log_path = Path(log_file_path).resolve()
        log_dir = log_path.parent
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            if not any(log_dir.iterdir()):
                logging.info(f"Created log directory: {log_dir}")
        
        # Create RotatingFileHandler
        file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Create StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        # Get the root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # Avoid adding multiple handlers
        if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
            logger.addHandler(file_handler)
        
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            logger.addHandler(stream_handler)
        
        logging.info(f"Logging configured. Log file: {log_path}")
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        raise


def load_config(config_file_path: Optional[Union[str, Path]] = None) -> dict:
    """
    Loads the YAML configuration file. If no path is provided, it defaults
    to `[base_path]/config/config.yaml`.

    Parameters:
    - config_file_path (str or Path, optional): Path to the YAML configuration file.
      If None, loads from 'config/config.yaml' relative to the base path.

    Returns:
    - dict: Configuration parameters.

    Raises:
    - FileNotFoundError: If the configuration file does not exist.
    - yaml.YAMLError: If there is an error parsing the YAML file.
    - KeyError: If required configuration keys are missing.
    """
    try:
        if config_file_path is None:
            base_path = get_base_path()
            config_file_path = base_path / "config" / "config.yaml"
        
        config_path = Path(config_file_path).resolve()
        with config_path.open('r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}")

        # Validate required keys for visualization
        required_keys = ['annotated_csv', 'combined_matrices_dir', 'plots_output_dir', 'diam_vel_mapping_file']
        viz_config = config.get('visualization', {})
        missing_keys = [key for key in required_keys if key not in viz_config or not viz_config[key]]
        if missing_keys:
            logging.critical(f"Missing required configuration parameters: {missing_keys}")
            raise KeyError(f"Missing required configuration parameters: {missing_keys}")

        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise
    except KeyError as e:
        logging.error(e)
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {e}")
        raise


def ensure_directory_exists(directory_path: Union[str, Path]) -> None:
    """
    Ensures that the specified directory exists. Creates it if it doesn't.

    Parameters:
    - directory_path (str or Path): Path to the directory.

    Raises:
    - OSError: If the directory cannot be created and does not already exist.
    """
    directory = Path(directory_path).resolve()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured that directory exists: {directory}")
    except OSError as e:
        logging.error(f"Failed to ensure directory exists: {directory}. Error: {e}")
        raise


def validate_directories(root_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Validates the existence of the root and output directories.
    Creates the output directory if it does not exist.

    Parameters:
    - root_dir (str or Path): Path to the root directory.
    - output_dir (str or Path): Path to the output directory.

    Raises:
    - FileNotFoundError: If root_dir does not exist.
    - NotADirectoryError: If root_dir or output_dir exists but is not a directory.
    """
    root = Path(root_dir).resolve()
    output = Path(output_dir).resolve()
    
    if not root.exists():
        logging.error(f"Root directory does not exist: {root}")
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    if not root.is_dir():
        logging.error(f"Root path is not a directory: {root}")
        raise NotADirectoryError(f"Root path is not a directory: {root}")
    
    # Validate and ensure output_dir exists
    ensure_directory_exists(output)


def find_files(root_directory: Union[str, Path],
               file_patterns: Optional[List[str]] = None,
               exclude_dirs: Optional[List[Union[str, Path]]] = None,
               exclude_files: Optional[List[str]] = None,
               exclude_file_patterns: Optional[List[str]] = None) -> List[str]:
    """
    Recursively find files within the root_directory matching file_patterns,
    excluding specified directories and files based on exact names or patterns.

    Parameters:
    - root_directory (str or Path): The root directory to search.
    - file_patterns (List[str], optional): List of glob patterns to match files (e.g., ['*.txt', '*.csv']).
    - exclude_dirs (List[str or Path], optional): List of directory paths to exclude from search.
    - exclude_files (List[str], optional): List of specific file names to exclude from search.
    - exclude_file_patterns (List[str], optional): List of glob patterns to exclude files (e.g., ['*temp*', '*.bak']).

    Returns:
    - List[str]: A list of file paths matching the file_patterns and not excluded.

    Raises:
    - FileNotFoundError: If the root_directory does not exist.
    - NotADirectoryError: If the root_directory is not a directory.
    """
    if file_patterns is None:
        file_patterns = ['*.txt']  # Default
    
    p = Path(root_directory).resolve()
    
    if not p.exists():
        logging.error(f"Root directory does not exist: {root_directory}")
        raise FileNotFoundError(f"Root directory does not exist: {root_directory}")
    if not p.is_dir():
        logging.error(f"Root path is not a directory: {root_directory}")
        raise NotADirectoryError(f"Root path is not a directory: {root_directory}")
    
    # Resolve exclusion paths
    exclude_dirs = [Path(ep).resolve() for ep in exclude_dirs] if exclude_dirs else []
    exclude_files = set(exclude_files) if exclude_files else set()
    exclude_file_patterns = exclude_file_patterns if exclude_file_patterns else []
    
    def is_excluded(file: Path) -> bool:
        # Check if the file is within any excluded directories
        if any(ex_path in file.parents for ex_path in exclude_dirs):
            return True
        # Check if the file name is in the excluded files
        if file.name in exclude_files:
            return True
        # Check if the file name matches any excluded patterns
        for pattern in exclude_file_patterns:
            if fnmatch.fnmatch(file.name, pattern):
                return True
        return False
    
    matched_files = []
    for pattern in file_patterns:
        for file in p.rglob(pattern):
            if file.is_file() and not is_excluded(file):
                matched_files.append(str(file))
    
    excluded_dirs_str = [str(ep) for ep in exclude_dirs]
    excluded_files_str = list(exclude_files)
    excluded_file_patterns_str = list(exclude_file_patterns)
    logging.info(
        f"Found {len(matched_files)} files in {root_directory} "
        f"matching patterns {file_patterns} "
        f"(excluding directories: {excluded_dirs_str}, files: {excluded_files_str}, "
        f"patterns: {excluded_file_patterns_str})"
    )
    return matched_files


def load_matrix(matrix_file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single matrix file (CSV or NPY) into a Pandas DataFrame.

    Parameters:
    - matrix_file_path (str or Path): Path to the matrix file.

    Returns:
    - pd.DataFrame: DataFrame with velocities as rows and diameters as columns.

    Raises:
    - FileNotFoundError: If the matrix file does not exist.
    - ValueError: If the file format is unsupported.
    - Exception: If loading the file fails.
    """
    matrix_path = Path(matrix_file_path).resolve()
    if not matrix_path.exists():
        logging.error(f"Matrix file does not exist: {matrix_path}")
        raise FileNotFoundError(f"Matrix file does not exist: {matrix_path}")
    
    try:
        if matrix_path.suffix.lower() == '.csv':
            df = pd.read_csv(matrix_path, index_col=0)
            df.index = df.index.astype(float)
            df.columns = df.columns.astype(float)
            logging.info(f"Loaded matrix from {matrix_path}")
            return df
        elif matrix_path.suffix.lower() == '.npy':
            matrix = np.load(matrix_path)
            df = pd.DataFrame(matrix)
            # Assuming that the matrix doesn't contain headers; adjust if necessary
            df.index = df.index.astype(float)
            df.columns = df.columns.astype(float)
            logging.info(f"Loaded matrix from {matrix_path}")
            return df
        else:
            logging.error(f"Unsupported file format for matrix: {matrix_path}")
            raise ValueError(f"Unsupported file format for matrix: {matrix_path}")
    except Exception as e:
        logging.error(f"Failed to load matrix from {matrix_path}: {e}", exc_info=True)
        raise


def load_aggregated_data(aggregated_file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the aggregated data CSV file into a Pandas DataFrame.

    Parameters:
    - aggregated_file_path (str or Path): Path to the aggregated data CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing aggregated data with 'Datetime' as datetime objects if present.

    Raises:
    - FileNotFoundError: If the aggregated data file does not exist.
    - ValueError: If the file format is unsupported.
    - Exception: If loading the file fails.
    """
    path = Path(aggregated_file_path).resolve()
    if not path.exists():
        logging.error(f"Aggregated data file does not exist: {path}")
        raise FileNotFoundError(f"Aggregated data file does not exist: {path}")
    if path.suffix.lower() != '.csv':
        logging.error(f"Unsupported file format for aggregated data: {path}")
        raise ValueError(f"Unsupported file format for aggregated data: {path}")
    
    try:
        df = pd.read_csv(path)
        if 'Datetime' in df.columns:
            try:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
            except Exception as e:
                logging.warning(f"Failed to parse 'Datetime' column in {path}: {e}")
        logging.info(f"Loaded aggregated data from {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load aggregated data from {path}: {e}", exc_info=True)
        raise


def parse_filename_to_date(filename: str) -> Optional[datetime]:
    """
    Parses the given filename and returns a datetime object extracted from it.

    Supported formats:

    1. Format like DSD2_YYYYMMDD_HHMMSS.txt  
    e.g., dsd2_data_20250101_000000.txt

    2. Format like VSTTEST_data_YYYYMMDD_HHMMSS.txt  
    e.g., VSTTEST_data_20250101_000000.txt

    3. Format like dsd1_pilar_data_YYYYMMDD_HHMMSS.txt  
    e.g., dsd1_pilar_data_20250102_150000.txt

    The function searches the underscore‑separated parts for an 8‑digit token (YYYYMMDD)
    and then uses the following token (with file extension stripped) as the time token.
    """
    name = Path(filename).name

    # Split the filename by underscores.
    parts = name.split('_')

    # Loop through tokens searching for a token that is exactly 8 digits.
    for i, token in enumerate(parts):
        token_clean = token.split('.')[0]  # Remove extension if present
        if len(token_clean) == 8 and token_clean.isdigit():
            date_str = token_clean  # Expect this to be YYYYMMDD
            # Look for a following token that might be the time.
            if i + 1 < len(parts):
                next_token = parts[i+1].split('.')[0]  # Remove file extension if any
                # Try as HHMMSS (6 digits).
                if len(next_token) == 6 and next_token.isdigit():
                    try:
                        return datetime.strptime(date_str + next_token, "%Y%m%d%H%M%S")
                    except ValueError as e:
                        logging.warning(f"Error parsing datetime from {date_str} and {next_token}: {e}")
                        return None
                # Otherwise, try as HHMM (4 digits) and assume seconds=00.
                elif len(next_token) == 4 and next_token.isdigit():
                    try:
                        return datetime.strptime(date_str + next_token, "%Y%m%d%H%M")
                    except ValueError as e:
                        logging.warning(f"Error parsing datetime from {date_str} and {next_token}: {e}")
                        return None
            # If no valid time token follows, break out.
            break

    # If neither format matched
    logging.warning(f"Filename {name} doesn't match expected patterns.")
    return None


def cleanup_output(config: dict) -> None:
    """
    Deletes specified directories and files based on the cleanup configuration.

    Parameters:
    - config (dict): The loaded configuration dictionary.

    Raises:
    - Exception: If deletion fails for any path.
    """
    cleanup_config = config.get('cleanup', {})
    directories: List[str] = cleanup_config.get('directories', [])
    files: List[str] = cleanup_config.get('files', [])

    # Delete directories
    for dir_path in directories:
        path = Path(dir_path).resolve()
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
                logging.info(f"Deleted directory: {path}")
            except Exception as e:
                logging.error(f"Failed to delete directory {path}: {e}")
                raise
        else:
            logging.info(f"Directory not found (skipping): {path}")

    # Delete files
    for file_path in files:
        path = Path(file_path).resolve()
        if path.exists() and path.is_file():
            try:
                path.unlink()
                logging.info(f"Deleted file: {path}")
            except Exception as e:
                logging.error(f"Failed to delete file {path}: {e}")
                raise
        else:
            logging.info(f"File not found (skipping): {path}")
