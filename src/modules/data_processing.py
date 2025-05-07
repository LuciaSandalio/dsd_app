# modules/data_processing.py

import re
import numpy as np
import pandas as pd
import datetime
import logging
from typing import List, Optional, Union, Tuple, Set, Dict
from pathlib import Path

# Import utility functions from utils.py
from modules.utils import ensure_directory_exists

# Constants
MATRIX_SIZE = 32
EXPECTED_MATRIX_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE  # 1024


def load_data(csv_file_path: Union[str, Path],
              required_columns: Optional[Set[str]] = None,
              dtype_spec: Optional[Dict[str, Union[str, type]]] = None) -> pd.DataFrame:
    """
    Load and prepare the data from a CSV file.

    Parameters:
    - csv_file_path (str or Path): Path to the input CSV file.
    - required_columns (set, optional): Set of required column names. If provided, the function will
                                       raise an error if any of these columns are missing.
    - dtype_spec (dict, optional): Dictionary specifying data types for columns, e.g., {'Timestamp': 'int64'}.

    Returns:
    - pd.DataFrame: Prepared DataFrame sorted by 'Datetime'.
    
    Raises:
    - FileNotFoundError: If the CSV file does not exist.
    - ValueError: If required columns are missing or if 'Datetime' column is absent.
    - pd.errors.EmptyDataError: If the CSV file is empty.
    - pd.errors.ParserError: If the CSV file cannot be parsed.
    - Exception: For any other unforeseen errors during loading.
    """
    csv_path = Path(csv_file_path).resolve()
    required_columns = required_columns or set()
    
    try:
        if not csv_path.exists():
            logging.error(f"CSV file does not exist: {csv_path}")
            raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
        
        df = pd.read_csv(csv_path, dtype=dtype_spec)
        
        if required_columns:
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                logging.error(f"Missing required columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        if 'Datetime' not in df.columns:
            logging.error("'Datetime' column is missing from the CSV file.")
            raise ValueError("'Datetime' column is required but missing.")
        
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.sort_values(by='Datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        logging.info("Data loaded successfully. Here's a preview:")
        logging.info(df.head())
        
        return df
    except FileNotFoundError as e:
        logging.error(e)
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"No data: The file at {csv_path} is empty.")
        raise
    except pd.errors.ParserError:
        logging.error(f"Parsing error: Could not parse the CSV file at {csv_path}.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        raise

def process_file(file_path: str) -> Tuple[List[List[float]], List[Tuple[int, np.ndarray]]]:
    """
    Process a single .txt file, extracting data fields and matrices.

    Parameters:
    - file_path (str): Path to the .txt file.

    Returns:
    - Tuple containing:
        - data_rows (List[List[float]]): Extracted data rows.
        - matrix_data (List[Tuple[int, np.ndarray]]): Tuples of timestamp and matrix.
    """
    data_rows: List[List[float]] = []
    matrix_data: List[Tuple[int, np.ndarray]] = []


    try:
        logging.info(f"Starting processing file: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                tokens = line.split(maxsplit=1)
                if len(tokens) < 2:
                    logging.warning(f"File '{file_path}', Line {line_num} - Incorrect format. Skipping line.")
                    continue
                first_token = tokens[0]
                if first_token.isdigit():
                    # ===== OLD FORMAT =====
                    try:
                        timestamp = int(first_token)
                    except ValueError:
                        logging.warning(f"File '{file_path}', Line {line_num} - Invalid Unix timestamp: {first_token}")
                        continue
                    dt_obj = datetime.datetime.fromtimestamp(timestamp)
                    formatted_time = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                    rest = tokens[1]
                    parts_rest = rest.split(":")
                    if len(parts_rest) < 10:
                        logging.warning(f"File '{file_path}', Line {line_num} - Insufficient data sections in old format.")
                        continue
                    data_section = parts_rest[:9]
                    matrix_string = parts_rest[9]
                    try:
                        data_fields = list(map(float, data_section))
                    except ValueError as e:
                        logging.error(f"File '{file_path}', Line {line_num} - Error converting data fields: {e}")
                        continue
                    matrix_string_clean = re.sub(r'\D', '', matrix_string)
                    if len(matrix_string_clean) != EXPECTED_MATRIX_ELEMENTS * 3:
                        logging.warning(f"File '{file_path}', Line {line_num} - Matrix length mismatch in old format. Expected {EXPECTED_MATRIX_ELEMENTS * 3} digits, got {len(matrix_string_clean)}.")
                        continue
                    matrix_cells = [int(matrix_string_clean[i:i+3]) for i in range(0, len(matrix_string_clean), 3)]
                    if len(matrix_cells) != EXPECTED_MATRIX_ELEMENTS:
                        logging.warning(f"File '{file_path}', Line {line_num} - Matrix elements count mismatch in old format.")
                        continue
                    matrix = np.array(matrix_cells, dtype=np.int32).reshape(MATRIX_SIZE, MATRIX_SIZE)
                    data_rows.append([timestamp, formatted_time] + data_fields)
                    if not np.all(matrix == 0):
                        matrix_data.append((timestamp, matrix))
                else:
                    # ===== NEW FORMAT =====
                    # Expect the first 19 characters to be the datetime string in format "YYYY-MM-DD HH:MM:SS"
                    if len(line) < 20:
                        logging.warning(f"File '{file_path}', Line {line_num} - Line too short for new format.")
                        continue
                    dt_str = line[:19]
                    try:
                        dt_obj = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError as e:
                        logging.error(f"File '{file_path}', Line {line_num} - Error parsing datetime in new format: {e}")
                        continue
                    timestamp = int(dt_obj.timestamp())
                    formatted_time = dt_str
                    # The rest (starting at character index 20) should contain semicolon-separated tokens.
                    rest_of_line = line[20:].strip()
                    # Split by semicolon.
                    tokens_new = [t.strip() for t in rest_of_line.split(";") if t.strip()]
                    matrix_tokens = []
                    extra_tokens = []
                    # Classify tokens: tokens that exactly match three digits are assumed part of the matrix.
                    for token in tokens_new:
                        if re.fullmatch(r"\d{3}", token):
                            matrix_tokens.append(token)
                        else:
                            extra_tokens.append(token)
                    if len(matrix_tokens) < EXPECTED_MATRIX_ELEMENTS:
                        logging.warning(f"File '{file_path}', Line {line_num} - Matrix token count mismatch in new format. Expected at least {EXPECTED_MATRIX_ELEMENTS} tokens, got {len(matrix_tokens)}.")
                        continue
                    # Use only the first 1024 tokens to build the matrix.
                    matrix_string_clean = "".join(matrix_tokens[:EXPECTED_MATRIX_ELEMENTS])
                    if len(matrix_string_clean) != EXPECTED_MATRIX_ELEMENTS * 3:
                        logging.warning(f"File '{file_path}', Line {line_num} - Matrix string length mismatch in new format.")
                        continue
                    matrix_cells = [int(matrix_string_clean[i:i+3]) for i in range(0, len(matrix_string_clean), 3)]
                    if len(matrix_cells) != EXPECTED_MATRIX_ELEMENTS:
                        logging.warning(f"File '{file_path}', Line {line_num} - Matrix elements count mismatch in new format after reassembly.")
                        continue
                    matrix = np.array(matrix_cells, dtype=np.int32).reshape(MATRIX_SIZE, MATRIX_SIZE)
                    # Process extra tokens, if any. Try converting them to float.
                    try:
                        extra_data = list(map(float, extra_tokens)) if extra_tokens else []
                    except ValueError as e:
                        logging.warning(f"File '{file_path}', Line {line_num} - Could not convert extra fields to float: {e}")
                        extra_data = []
                    data_rows.append([timestamp, formatted_time] + extra_data)
                    if not np.all(matrix == 0):
                        matrix_data.append((timestamp, matrix))
                # End if-else for format
            # End for each line
    except Exception as e:
        logging.error(f"Failed to process file '{file_path}': {e}", exc_info=True)
    return data_rows, matrix_data

def save_dataframe(data_rows: List[List[float]], output_csv_path: str) -> None:
    """
    Save the extracted data rows to a CSV file, normalizing both:
      - OLD rows: 9 data fields → [Nº de partículas, Temp Sensor, …, Estado Sensor]
      - NEW rows: 4 data fields → [Intensidad máxima, Precipitación acumulada,
                                  Factor de reflectividad, Estado del sensor]
    into the SAME 9‑field layout so that
    'Nº de partículas' and 'Intensidad' always exist.
    """

    if not data_rows:
        logging.warning("No data rows to save.")
        return

    # The *uniform* column schema we want:
    columns = [
        "Timestamp", "Datetime",
        "Nº de partículas", "Temp Sensor",
        "Temp cabezal derecho", "Temp cabezal izquierdo",
        "Corriente Calentador", "Intensidad",
        "Precipitacion", "Reflectividad", "Estado Sensor"
    ]

    normalized = []
    for row in data_rows:
        # OLD format: row == [ts, dt,  9 floats]
        if len(row) == 11:
            normalized.append(row)
        # NEW format: row == [ts, dt,  4 floats]
        elif len(row) == 6:
            ts, dt, i_max, prec_acu, refl, estado = row
            # pad the five missing old‑fields with NaN, then map:
            #   intensidad máxima → 'Intensidad'
            #   precip acumulada → 'Precipitacion'
            #   factor reflectividad → 'Reflectividad'
            #   estado del sensor → 'Estado Sensor'
            pad = [float('nan')] * 5
            full_data = pad + [i_max, prec_acu, refl, estado]
            normalized.append([ts, dt] + full_data)
        else:
            logging.warning(f"Skipping row of unexpected length {len(row)}: {row!r}")

    df = pd.DataFrame(normalized, columns=columns)

    # Convert, sort, reset
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.sort_values(by='Datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Chunked write
    file_exists = Path(output_csv_path).exists()
    chunk = 100_000
    for start in range(0, len(df), chunk):
        df.iloc[start:start+chunk].to_csv(
            output_csv_path,
            mode='a',
            header=not file_exists and start == 0,
            index=False
        )

    logging.info(f"DataFrame saved to {output_csv_path}")


def save_matrices(matrix_data: List[Tuple[int, np.ndarray]], output_directory: str) -> None:
    """
    Save matrix data to binary files.

    Parameters:
    - matrix_data (List[Tuple[int, np.ndarray]]): List of tuples containing timestamp and matrix.
    - output_directory (str): Directory to save the matrix files.

    Raises:
    - Exception: If there is an error during saving matrices.
    """
    try:
        # Create subdirectory for matrices
        matrices_dir = Path(output_directory) / "matrices"
        ensure_directory_exists(str(matrices_dir))
        logging.info(f"Matrices will be saved to {matrices_dir}")

        # Sort matrix_data by timestamp in ascending order
        matrix_data_sorted = sorted(matrix_data, key=lambda x: x[0])

        for timestamp, matrix in matrix_data_sorted:
            # Save matrix as binary file
            matrix_path = matrices_dir / f"matrix_{timestamp}.npy"
            np.save(matrix_path, matrix)
            logging.debug(f"Matrix saved to {matrix_path}")
        
        logging.info(f"All matrices have been saved to {matrices_dir}")
    except Exception as e:
        logging.error(f"Failed to save matrices to {output_directory}: {e}", exc_info=True)
        raise
