# modules/event_identification.py

import pandas as pd
import numpy as np
import logging
import shutil
import os
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict
from typing import Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed  
from tqdm import tqdm  # For progress visualization
from datetime import datetime
from modules.utils import ensure_directory_exists


def save_annotated_data(df: pd.DataFrame, annotated_csv_path: Union[str, Path]) -> None:
    """
    Save the annotated DataFrame with precipitation event identifiers to a CSV file.
    Backs up the existing file by appending a timestamped '.bak' extension.

    Parameters:
    - df (pd.DataFrame): Annotated DataFrame. Must contain 'Precip_Event' column.
    - annotated_csv_path (str or Path): Path to save the annotated CSV.

    Raises:
    - ValueError: If required columns are missing from the DataFrame.
    - Exception: If saving the DataFrame fails.
    """
    annotated_path = Path(annotated_csv_path).resolve()
    required_columns = {'Precip_Event'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logging.error(f"DataFrame is missing required columns for saving: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    logging.info(f"Columns before saving: {df.columns.tolist()}")
    logging.debug("Preview of annotated data:")
    logging.debug(df.head())

    try:
        # Backup existing file if it exists with a timestamp
        if annotated_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = annotated_path.with_suffix(f"{annotated_path.suffix}.bak.{timestamp}")
            shutil.copy(str(annotated_path), str(backup_path))
            logging.info(f"Backup of existing file created at {backup_path}")

        # Save the annotated DataFrame
        df.to_csv(annotated_path, index=False)
        logging.info(f"Annotated data saved to {annotated_path}")
    except Exception as e:
        logging.error(f"Failed to save annotated data to {annotated_csv_path}: {e}", exc_info=True)
        raise


def save_event(event_number: int, group: pd.DataFrame, event_dir: Union[str, Path]) -> None:
    """
    Save a single precipitation event to a CSV file.

    Parameters:
    - event_number (int): Unique identifier for the precipitation event.
    - group (pd.DataFrame): DataFrame containing the event data. Must include all relevant columns.
    - event_dir (str or Path): Directory to save the event CSV.

    Raises:
    - Exception: If saving the event fails.
    """
    event_directory = Path(event_dir).resolve()
    event_csv_path = event_directory / f"event_{event_number}.csv"
    
    if group.empty:
        logging.warning(f"Attempted to save event {event_number}, but the data group is empty.")
        return
    
    try:
        group.to_csv(event_csv_path, index=False)
        logging.info(f"Event {event_number} saved to {event_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save event {event_number} to {event_csv_path}: {e}", exc_info=True)
        raise


def extract_and_save_events(df: pd.DataFrame, 
                            event_dir: Union[str, Path], 
                            max_workers: Optional[int] = None) -> None:
    """
    Extract individual precipitation events from the annotated DataFrame and save each as a separate CSV file in parallel.

    Parameters:
    - df (pd.DataFrame): Annotated DataFrame with a 'Precip_Event' column indicating event numbers.
    - event_dir (str or Path): Directory to save individual event CSVs.
    - max_workers (int, optional): Maximum number of worker processes for parallel saving. Defaults to min(32, os.cpu_count() + 4).

    Raises:
    - ValueError: If required columns are missing from the DataFrame.
    - Exception: If saving any event fails.
    """
    required_columns = {'Precip_Event'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logging.error(f"DataFrame is missing required columns for event extraction: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows without event identifiers
    event_df = df.dropna(subset=['Precip_Event']).copy()
    event_df['Precip_Event'] = event_df['Precip_Event'].astype(int)
    grouped_events = event_df.groupby('Precip_Event')

    # Ensure the event directory exists
    ensure_directory_exists(event_dir)

    # Determine number of workers
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)

    # Parallelize saving individual events using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(save_event, event_number, group, event_dir): event_number
            for event_number, group in grouped_events
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving events", unit="event"):
            event_number = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error saving event {event_number}: {e}", exc_info=True)


def remove_duplicates(df: pd.DataFrame, subset_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows based on specified subset columns, ensuring the uniqueness of 'Datetime'.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - subset_columns (list, optional): Columns to consider for identifying duplicates. Defaults to ['Timestamp', 'Datetime'].

    Returns:
    - pd.DataFrame: DataFrame without duplicate rows based on subset columns.
    - int: Number of duplicates removed.

    Raises:
    - ValueError: If specified subset columns are missing from the DataFrame.
    """
    if subset_columns is None:
        subset_columns = ['Timestamp', 'Datetime']
    
    required_columns = set(subset_columns)
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logging.error(f"Missing required columns for duplicate removal: {missing}")
        raise ValueError(f"Missing required columns: {missing}")
    
    initial_count = len(df)
    df = df.drop_duplicates(subset=subset_columns, keep='first').reset_index(drop=True)
    final_count = len(df)
    duplicates_removed = initial_count - final_count
    logging.info(f"Removed {duplicates_removed} duplicate rows based on columns {subset_columns}.")

    # Ensure 'Datetime' is unique
    if df['Datetime'].duplicated().any():
        logging.warning("Warning: 'Datetime' column has duplicate entries after removing duplicates.")
    else:
        logging.info("'Datetime' column is unique.")

    return df, duplicates_removed

def ensure_continuous_timestamps(df: pd.DataFrame, default_freq: str = 'T') -> Tuple[pd.DataFrame, pd.DatetimeIndex, str]:
    """
    Ensure that the DataFrame has continuous timestamps based on the detected frequency.
    Marks missing data points and returns relevant information.

    Parameters:
    - df (pd.DataFrame): Input DataFrame. Must contain 'Datetime' and 'Timestamp' columns.
    - default_freq (str): Default frequency to use if inferring fails. Defaults to 'T' (minute).

    Returns:
    - pd.DataFrame: Reindexed DataFrame with continuous 'Datetime' timestamps.
    - pd.DatetimeIndex: Timestamps that were missing.
    - str: Detected or defaulted frequency of the data.

    Raises:
    - ValueError: If required columns are missing.
    """
    required_columns = {'Datetime', 'Timestamp'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logging.error(f"Missing required columns for timestamp continuity: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    # Detect frequency
    detected_freq = pd.infer_freq(df['Datetime'])
    if detected_freq is None:
        # If frequency cannot be inferred, use default_freq
        expected_freq = default_freq
        logging.info(f"Frequency could not be inferred. Defaulting to '{expected_freq}' (minute).")
    else:
        expected_freq = detected_freq
        logging.info(f"Inferred frequency: '{expected_freq}'.")

    # Set 'Datetime' as the index
    df.set_index('Datetime', inplace=True)

    # Check if 'Datetime' index is unique
    if not df.index.is_unique:
        logging.warning("'Datetime' index contains duplicate entries. Duplicates will be dropped before reindexing.")
        df = df[~df.index.duplicated(keep='first')]

    # Create a complete timestamp range based on the data's start and end
    full_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)

    # Reindex the DataFrame to include all timestamps, explicitly mark missing data
    df = df.reindex(full_time_index)
    df['MissingData'] = df['Timestamp'].isna()

    # Rename the index to 'Datetime' and reset it
    df.index.name = 'Datetime'
    df.reset_index(inplace=True)

    # Identify missing timestamps
    missing_timestamps = df[df['MissingData']]['Datetime']
    missing_count = len(missing_timestamps)
    ts_list = missing_timestamps.dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    logging.info(f"Number of missing timestamps: {missing_count}")

    if missing_count > 0:
        #logging.info("Missing timestamps detected:")
        #logging.debug(missing_timestamps)
        logging.info(f"Missing timestamps:{missing_timestamps}")

        # write the full list out to a separate file for review
        try:
            from pathlib import Path
            # assume your project root is two levels up from this module
            log_dir = Path(__file__).resolve().parents[2] / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            missing_file = log_dir / 'missing_timestamps.txt'
            with missing_file.open('w') as f:
                for ts in ts_list:
                    f.write(ts + '\n')
            logging.info(f"Saved full missing timestamps to {missing_file}")
        except Exception as e:
            logging.error(f"Failed to write missing timestamps file: {e}")
    else:
        logging.info("No missing timestamps detected.")

    return df, missing_timestamps, expected_freq

def mark_precipitation_activity(df: pd.DataFrame,
                                intensidad_threshold: float = 0.0
                                ) -> pd.DataFrame:
    """
    Mark rows where precipitation is active based on 'Intensidad' and 'Nº de partículas', excluding missing data.

    Parameters:
    - df (pd.DataFrame): Input DataFrame. Must contain 'Intensidad', 'Nº de partículas', and 'MissingData' columns.
    - intensidad_threshold (float): Minimum 'Intensidad' to consider precipitation active.
    - particulas_threshold (int): Minimum 'Nº de partículas' to consider precipitation active.

    Returns:
    - pd.DataFrame: DataFrame with an added 'Precip_Active' boolean column indicating active precipitation.
    
    Raises:
    - ValueError: If required columns are missing from the DataFrame.
    """
    required_columns = {'Intensidad', 'MissingData'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logging.error(f"Missing required columns for marking precipitation activity: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    df['Precip_Active'] = (
        (df['Intensidad'] > intensidad_threshold) &
        (~df['MissingData'])  # Ensure the row is not marked as missing data
    )
    logging.info(
        f"Marked precipitation activity with thresholds - Intensidad: >{intensidad_threshold}, excluding missing data."
    )

    # Additional logging for active precipitation points
    active_count = df['Precip_Active'].sum()
    logging.info(f"Number of precipitation-active points: {active_count}")

    if active_count == 0:
        logging.warning("No precipitation-active points detected based on the current thresholds.")
    elif active_count == len(df):
        logging.warning("All data points are marked as precipitation-active based on the current thresholds.")
    else:
        logging.info(f"Precipitation-active points range from {df['Datetime'].min()} to {df['Datetime'].max()}.")

    return df


def combine_matrices_for_event(
    event_timestamps: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    matrix_directory: Union[str, Path] = 'data/processed/matrices',
    output_csv_dir: Optional[Union[str, Path]] = None,
    diameters: Optional[List[float]] = None,
    velocities: Optional[List[float]] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load and combine matrices based on event timestamps and/or a date range.

    Parameters:
    - event_timestamps (Optional[List[int]]): List of UNIX timestamps for specific events.
    - start_date (Optional[str]): Start date in 'YYYY-MM-DD' format.
    - end_date (Optional[str]): End date in 'YYYY-MM-DD' format.
    - matrix_directory (str or Path): Directory where matrix files are stored.
    - output_csv_dir (Optional[Union[str, Path]]): Full path to save the output CSV file.
    - diameters (Optional[List[float]]): List of diameters for matrix columns.
    - velocities (Optional[List[float]]): List of velocities for matrix rows.

    Returns:
    - Optional[Dict[str, np.ndarray]]:
        - For event timestamps: {'combined_event_matrix': np.ndarray}
        - For date range: {'date_range_matrix': np.ndarray}
        Returns None if no matrices are loaded.
    """
    try:
        matrix_dir = Path(matrix_directory).resolve()
        output_path = Path(output_csv_dir).resolve() if output_csv_dir else None
        event_matrices = []
        date_range_matrices = []
        expected_shape = None
        loaded_timestamps = set()

        # Validate input identifiers
        if not event_timestamps and not (start_date and end_date):
            logging.error("Insufficient identifiers provided. Provide either event_timestamps, a date range, or both.")
            return None

        # Create output directory if saving CSVs
        if output_path:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                logging.debug(f"Ensured existence of directory: {output_path.parent}")
            except Exception as e:
                logging.error(f"Failed to create directory {output_path.parent}: {e}", exc_info=True)
                return None

        # Collect matrices based on event_timestamps
        if event_timestamps:
            for timestamp in event_timestamps:
                timestamp_int = int(timestamp)
                if timestamp_int in loaded_timestamps:
                    logging.debug(f"Matrix for timestamp {timestamp_int} already loaded. Skipping to avoid double-counting.")
                    continue
                matrix_path = matrix_dir / f"matrix_{timestamp_int}.npy"
                if matrix_path.exists():
                    try:
                        matrix = np.load(matrix_path)
                        if expected_shape is None:
                            expected_shape = matrix.shape
                            logging.debug(f"Expected matrix shape set to: {expected_shape}")
                        elif matrix.shape != expected_shape:
                            logging.error(f"Inconsistent matrix shapes: {matrix.shape} vs {expected_shape}")
                            raise ValueError(f"Inconsistent matrix shapes: {matrix.shape} vs {expected_shape}")
                        event_matrices.append(matrix)
                        loaded_timestamps.add(timestamp_int)
                        logging.debug(f"Loaded matrix from {matrix_path}")
                    except Exception as e:
                        logging.error(f"Error loading matrix {matrix_path}: {e}", exc_info=True)
                        continue
                else:
                    logging.warning(f"Matrix file for timestamp {timestamp_int} not found at {matrix_path}.")

            if event_matrices:
                combined_event_matrix = np.sum(event_matrices, axis=0)
                logging.debug(f"Combined event matrix shape: {combined_event_matrix.shape}")

                # Save combined event matrix as CSV if required
                if output_path:
                    try:
                        df_combined = pd.DataFrame(combined_event_matrix)
                        # Add headers if diameters and velocities are provided
                        if diameters is not None:
                            if len(diameters) != combined_event_matrix.shape[1]:
                                logging.error("Number of diameters does not match matrix columns.")
                                raise ValueError("Number of diameters does not match matrix columns.")
                            df_combined.columns = [f"{d}mm" for d in diameters]
                            logging.debug("Added diameter headers to combined event matrix.")
                        if velocities is not None:
                            if len(velocities) != combined_event_matrix.shape[0]:
                                logging.error("Number of velocities does not match matrix rows.")
                                raise ValueError("Number of velocities does not match matrix rows.")
                            df_combined.index = [f"{v}m/s" for v in velocities]
                            logging.debug("Added velocity indices to combined event matrix.")
                        df_combined.to_csv(output_path, index=(velocities is not None), header=(diameters is not None))
                        logging.info(f"Combined event matrix saved as CSV at {output_path}")
                    except Exception as e:
                        logging.error(f"Failed to save combined event matrix as CSV at {output_path}: {e}", exc_info=True)

                # Prepare output dictionary
                output = {'combined_event_matrix': combined_event_matrix}
            else:
                logging.info("No matrices found for the specified event timestamps.")
                output = {}
        else:
            output = {}

        # Collect matrices based on date range
        if start_date and end_date:
            try:
                # Convert start_date and end_date to datetime objects
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')

                # Ensure start_date <= end_date
                if start_dt > end_dt:
                    logging.error("start_date must be earlier than or equal to end_date.")
                    return None

            except ValueError as ve:
                logging.error(f"Invalid date format: {ve}")
                return None

            # Convert dates to UNIX timestamps (integer)
            start_timestamp = int(start_dt.timestamp())
            end_timestamp = int(end_dt.timestamp())

            # Collect all relevant matrix files within the date range
            try:
                all_files = matrix_dir.glob('matrix_*.npy')
                for file in all_files:
                    try:
                        # Extract timestamp from filename
                        parts = file.stem.split('_')
                        if len(parts) < 2:
                            raise ValueError("Filename does not contain timestamp.")
                        timestamp_str = parts[1]
                        timestamp = int(timestamp_str)
                        if start_timestamp <= timestamp <= end_timestamp:
                            if timestamp in loaded_timestamps:
                                logging.debug(f"Matrix for timestamp {timestamp} already loaded. Skipping to avoid double-counting.")
                                continue
                            matrix = np.load(file)
                            if expected_shape is None:
                                expected_shape = matrix.shape
                                logging.debug(f"Expected matrix shape set to: {expected_shape}")
                            elif matrix.shape != expected_shape:
                                logging.error(f"Inconsistent matrix shapes: {matrix.shape} vs {expected_shape}")
                                raise ValueError(f"Inconsistent matrix shapes: {matrix.shape} vs {expected_shape}")
                            date_range_matrices.append(matrix)
                            loaded_timestamps.add(timestamp)
                            logging.debug(f"Loaded matrix from {file}")

                    except (IndexError, ValueError) as parse_error:
                        logging.warning(f"Filename {file.name} does not match expected pattern: {parse_error}")
                        continue

                if date_range_matrices:
                    date_range_matrix = np.sum(date_range_matrices, axis=0)
                    logging.debug(f"Combined date range matrix shape: {date_range_matrix.shape}")

                    # Save date range matrix as CSV if required
                    if output_path:
                        try:
                            df_date_range = pd.DataFrame(date_range_matrix)
                            # Add headers if diameters and velocities are provided
                            if diameters is not None:
                                if len(diameters) != date_range_matrix.shape[1]:
                                    logging.error(f"Number of diameters ({len(diameters)}) does not match matrix columns ({date_range_matrix.shape[1]}).")
                                    raise ValueError(f"Number of diameters ({len(diameters)}) does not match matrix columns ({date_range_matrix.shape[1]}).")
                                df_date_range.columns = [f"{d}mm" for d in diameters]
                                logging.debug("Added diameter headers to date range matrix.")
                            if velocities is not None:
                                if len(velocities) != date_range_matrix.shape[0]:
                                    logging.error(f"Number of velocities ({len(velocities)}) does not match matrix rows ({date_range_matrix.shape[0]}).")
                                    raise ValueError(f"Number of velocities ({len(velocities)}) does not match matrix rows ({date_range_matrix.shape[0]}).")
                                df_date_range.index = [f"{v}m/s" for v in velocities]
                                logging.debug("Added velocity indices to date range matrix.")

                            # Directly use output_path as the target CSV file
                            df_date_range.to_csv(output_path, index=(velocities is not None), header=(diameters is not None))
                            logging.info(f"Date range matrix saved as CSV at {output_path}")
                        except Exception as e:
                            logging.error(f"Failed to save date range matrix as CSV at {output_path}: {e}", exc_info=True)

                    # Update output dictionary
                    output['date_range_matrix'] = date_range_matrix
                else:
                    logging.info("No matrices found within the specified date range.")
            except Exception as e:
                logging.error(f"Error accessing matrix files in {matrix_dir}: {e}", exc_info=True)

        # Final output check
        if not output:
            logging.info("No matrices were loaded based on the provided inputs.")
            return None

        return output

    except Exception as e:
        logging.error(f"Error combining matrices for event/date range: {e}", exc_info=True)
        return None


    
def identify_precipitation_events(df: pd.DataFrame,
                                  min_gap_hours: int = 2,
                                  matrix_directory: Union[str, Path] = "matrices",
                                  combined_matrix_directory: Union[str, Path] = "combined_matrices") -> Tuple[pd.DataFrame, int]:
    """
    Identify and assign precipitation events based on a minimum gap between active precipitation points.
    Combines corresponding matrices for each identified event.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a 'Precip_Active' boolean column.
    - min_gap_hours (int): Minimum gap in hours to consider the start of a new event.
    - matrix_directory (str or Path): Directory containing individual matrix files named as 'matrix_{timestamp}.npy'.
    - combined_matrix_directory (str or Path): Directory to save combined event matrices.

    Returns:
    - pd.DataFrame: DataFrame with an added 'Precip_Event' column indicating event numbers.
    - int: Total number of precipitation events identified.

    Raises:
    - ValueError: If required columns are missing or matrix files have inconsistent formats.
    """
    required_columns = {'Precip_Active', 'Datetime'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logging.error(f"Missing required columns for event identification: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    MIN_GAP = pd.Timedelta(hours=min_gap_hours)

    # Extract precipitation-active timestamps
    precip_timestamps = df.loc[df['Precip_Active'], 'Datetime'].reset_index(drop=True)

    if precip_timestamps.empty:
        logging.info("No precipitation events detected based on the current criteria.")
        df['Precip_Event'] = np.nan
        return df, 0

    # Calculate time differences between consecutive precipitation-active points
    time_gaps = precip_timestamps.diff()

    # Identify new events where the gap is greater than or equal to the specified minimum gap
    new_event_flags = time_gaps >= MIN_GAP

    # Assign event numbers using the cumulative sum of new event flags
    event_numbers = new_event_flags.cumsum().fillna(0).astype(int) + 1

    # Initialize 'Precip_Event' column
    df['Precip_Event'] = np.nan

    # Assign event numbers to precipitation-active rows
    df.loc[df['Precip_Active'], 'Precip_Event'] = event_numbers.values

    # Forward fill to propagate event numbers, but only within active precipitation
    df['Precip_Event'] = df['Precip_Event'].ffill().where(df['Precip_Active'])

    # Correct event count
    event_count = df['Precip_Event'].nunique() - (1 if df['Precip_Event'].isna().any() else 0)
    logging.info(f"Total precipitation events identified: {event_count}")

    # Debug: Print some event assignments
    logging.info("Sample precipitation event assignments:")
    logging.debug(df[['Datetime', 'Intensidad', 'Nº de partículas', 'Precip_Event']].dropna().head())

    unique_events = df['Precip_Event'].dropna().unique()
    combined_matrix_dir = Path(combined_matrix_directory).resolve()
    ensure_directory_exists(combined_matrix_dir)

    for event_id in unique_events:
        # Convert pd.Timestamp to Unix timestamp or appropriate format based on matrix naming convention
        event_timestamps = df[df['Precip_Event'] == event_id]['Timestamp'].astype(int) 

        # Logging the list of timestamps for the event
        logging.info(f"Processing Event ID: {event_id}")
        logging.info(f"Timestamps for Event {event_id}: {event_timestamps}")
        
        # Optionally, print to console
        print(f"Processing Event ID: {event_id}")
        print(f"Timestamps for Event {event_id}: {event_timestamps}")

        event_dir = Path(combined_matrix_directory)
        # 1) define two sub‑folders
        csv_dir = event_dir / "combined_csv"
        npy_dir = event_dir / "combined_npy"

        # 2) make sure they exist
        csv_dir.mkdir(parents=True, exist_ok=True)
        npy_dir.mkdir(parents=True, exist_ok=True)

        # 3) build the output paths
        csv_out = csv_dir / f"combined_event_{event_id}.csv"
        npy_out = npy_dir / f"combined_event_{event_id}.npy"

        result = combine_matrices_for_event(
         event_timestamps=event_timestamps.tolist(),
        matrix_directory=matrix_directory,
        output_csv_dir=str(csv_out)
        )
        if result and 'combined_event_matrix' in result:
            np.save(npy_out, result['combined_event_matrix'])
            logging.info(f"Combined matrix saved to {npy_out}")
        else:
            logging.warning(f"No matrices combined for Event {event_id}")

    return df, event_count

def validate_results(df: pd.DataFrame, event_count: int) -> None:
    """
    Validate the results of precipitation event identification.

    Parameters:
    - df (pd.DataFrame): Annotated DataFrame with 'Precip_Event' and 'Precip_Active' columns.
    - event_count (int): Total number of precipitation events identified.
    """
    required_columns = {'Precip_Event', 'Precip_Active', 'Datetime'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logging.error(f"DataFrame is missing required columns for validation: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    # Check if event_count is logical
    if event_count < 1:
        logging.info("No precipitation events identified.")
    else:
        logging.info(f"Successfully identified {event_count} precipitation event(s).")

    # Check for overlapping events (shouldn't happen)
    overlapping = df[(df['Precip_Event'].duplicated(keep=False)) & (df['Precip_Active'])]
    if not overlapping.empty:
        overlapping_events = overlapping['Precip_Event'].unique()
        logging.warning(f"Warning: Overlapping detected in events: {overlapping_events}")
        logging.warning(overlapping[['Datetime', 'Precip_Event']])
    else:
        logging.info("No overlapping events detected.")

    # Additional validation checks can be added here

