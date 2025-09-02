# modules/data_processing.py

import re
import os
import numpy as np, os, tempfile
import pandas as pd
import datetime
import logging
from typing import List, Optional, Union, Tuple, Set, Dict, Any
from pathlib import Path

# Import utility functions from utils.py
from modules.utils import ensure_directory_exists

# ------------------------------------------------------------------------------
# Helpers / configuration adapters
# ------------------------------------------------------------------------------

def _normalize_matrix_size(ms: Union[int, List[int], Tuple[int, int]]) -> int:
    """
    Accepts [rows, cols] or int; returns a single int (requires square).
    """
    if isinstance(ms, (list, tuple)) and len(ms) == 2:
        r, c = int(ms[0]), int(ms[1])
        if r != c:
            raise ValueError(f"Non-square matrix size {r}x{c} is not supported")
        return r
    return int(ms)

def _expected_elements(matrix_size: int) -> int:
    return int(matrix_size) * int(matrix_size)

def _coerce_datetime(df: pd.DataFrame, tz: Optional[str] = None) -> pd.DataFrame:
    """
    Parse 'Datetime' column and optionally localize/convert to tz.
    """
    if "Datetime" not in df.columns:
        return df
    dt = pd.to_datetime(df["Datetime"], errors="coerce", utc=False)
    if tz:
        # If naive, localize; if already tz-aware, convert
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(tz)
        else:
            dt = dt.dt.tz_convert(tz)
    df["Datetime"] = dt
    return df

def apply_qc(df: pd.DataFrame, qc_cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    Optional QC filter. Call from scripts if desired.
    """
    if not qc_cfg:
        return df
    out = df

    if "min_particles" in qc_cfg and "Nº de partículas" in out.columns:
        out = out[out["Nº de partículas"] >= int(qc_cfg["min_particles"])]

    if "valid_sensor_states" in qc_cfg and "Estado Sensor" in out.columns:
        valid = set(qc_cfg["valid_sensor_states"])
        out = out[out["Estado Sensor"].isin(valid)]

    clip = qc_cfg.get("clip", {})
    if "diametro_mm" in clip and "Diametro" in out.columns:
        lo, hi = clip["diametro_mm"]
        out["Diametro"] = out["Diametro"].clip(lower=lo, upper=hi)
    if "velocidad_ms" in clip and "Velocidad" in out.columns:
        lo, hi = clip["velocidad_ms"]
        out["Velocidad"] = out["Velocidad"].clip(lower=lo, upper=hi)

    na_policy = qc_cfg.get("na_policy", "drop_row")
    if na_policy == "drop_row":
        out = out.dropna()
    elif na_policy == "impute_zero":
        out = out.fillna(0)
    # 'flag_only' -> leave as is
    return out

def dedup_and_sort_output_csv(csv_path: Union[str, Path], tz: Optional[str] = None) -> None:
    """
    Drop duplicate rows using ['Timestamp','Datetime'] if present,
    then sort stably. Prefer sorting by Timestamp (numeric), and rebuild
    Datetime from Timestamp in a single timezone to avoid mixed tz issues.
    """
    p = Path(csv_path)
    if not p.exists():
        return

    try:
        df = pd.read_csv(p)
    except Exception as e:
        logging.error(f"Could not read CSV for dedup: {p} -> {e}")
        return

    # Dedup
    subset = [c for c in ("Timestamp", "Datetime") if c in df.columns]
    if subset:
        df.drop_duplicates(subset=subset, keep="last", inplace=True)
    else:
        df.drop_duplicates(keep="last", inplace=True)

    # Sort & normalize Datetime
    if "Timestamp" in df.columns and pd.api.types.is_numeric_dtype(df["Timestamp"]):
        # Sort by numeric epoch
        df.sort_values("Timestamp", inplace=True)
        # Rebuild Datetime consistently from Timestamp
        dt = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
        if tz:
            dt = dt.dt.tz_convert(tz)
        df["Datetime"] = dt
    elif "Datetime" in df.columns:
        # Fallback: coerce and normalize timezone
        dt = pd.to_datetime(df["Datetime"], errors="coerce")
        # If series is tz-naive, localize; if tz-aware, convert
        if tz:
            if getattr(dt.dt, "tz", None) is None:
                dt = dt.dt.tz_localize(tz)
            else:
                dt = dt.dt.tz_convert(tz)
        df["Datetime"] = dt
        df.sort_values("Datetime", inplace=True)

    df.to_csv(p, index=False)
    logging.info(f"Deduplicated, normalized timezone and sorted CSV: {p}")


# ------------------------------------------------------------------------------
# Safe-write helpers
# ------------------------------------------------------------------------------

def _safe_write_csv(df: pd.DataFrame, dest: Union[str, Path], safe: bool = True) -> None:
    dest = Path(dest)
    ensure_directory_exists(dest.parent)
    if safe:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(dest)
    else:
        df.to_csv(dest, index=False)

def _safe_write_npy(arr, dest, safe=True):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)  # <- ensure folder exists

    if not safe:
        with open(dest, "wb") as f:
            np.save(f, arr)
        return str(dest)

    # write tmp IN THE SAME DIRECTORY for atomic replace
    with tempfile.NamedTemporaryFile(
        dir=dest.parent, prefix=dest.name + ".", suffix=".tmp", delete=False
    ) as tf:
        np.save(tf, arr)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_path = Path(tf.name)

    tmp_path.replace(dest)
    return str(dest)
    

# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_data(csv_file_path: Union[str, Path],
              required_columns: Optional[Set[str]] = None,
              dtype_spec: Optional[Dict[str, Union[str, type]]] = None,
              tz: Optional[str] = None) -> pd.DataFrame:
    """
    Load and prepare the data from a CSV file.

    Parameters:
    - csv_file_path: path to the input CSV file.
    - required_columns: set of required column names; error if any are missing.
    - dtype_spec: dict of column dtypes (e.g., {'Timestamp': 'int64'}).
    - tz: optional timezone to localize/convert the 'Datetime' column.

    Returns:
    - DataFrame sorted by 'Datetime'.
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

        if "Datetime" in df.columns:
            df = _coerce_datetime(df, tz=tz)
            df.sort_values(by="Datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
        elif "Timestamp" in df.columns:
            # Fallback: sort by Timestamp if Datetime is absent
            df.sort_values(by="Timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df
    except FileNotFoundError:
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

# ------------------------------------------------------------------------------
# Raw file → rows + matrices
# ------------------------------------------------------------------------------

def process_file(file_path: str,
                 matrix_size: int = 32) -> Tuple[List[List[float]], List[Tuple[int, np.ndarray]]]:
    """
    Process a single .txt file, extracting data fields and matrices.

    Parameters:
    - file_path: path to the .txt file.
    - matrix_size: side of the (square) DSD matrix (default 32).

    Returns:
    - (data_rows, matrix_data):
        * data_rows: List of [timestamp:int, formatted_time:str] + data fields
        * matrix_data: List of (timestamp:int, matrix:np.ndarray[matrix_size x matrix_size])
    """
    data_rows: List[List[float]] = []
    matrix_data: List[Tuple[int, np.ndarray]] = []
    expected_elems = _expected_elements(matrix_size)

    try:
        logging.info(f"Starting processing file: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                # Split first token to detect "old" vs "new" format
                tokens = line.split(maxsplit=1)
                if len(tokens) < 2:
                    logging.warning(f"{file_path}, line {line_num}: incorrect format (skipping).")
                    continue

                first_token = tokens[0]
                if first_token.isdigit():
                    # ---------------- OLD FORMAT ----------------
                    # first token is Unix timestamp (seconds)
                    try:
                        timestamp = int(first_token)
                    except ValueError:
                        logging.warning(f"{file_path}, line {line_num}: invalid Unix timestamp '{first_token}'.")
                        continue

                    dt_obj = datetime.datetime.fromtimestamp(timestamp)
                    formatted_time = dt_obj.strftime("%Y-%m-%d %H:%M:%S")

                    rest = tokens[1]
                    parts_rest = rest.split(":")
                    if len(parts_rest) < 10:
                        logging.warning(f"{file_path}, line {line_num}: insufficient sections for old format.")
                        continue

                    data_section = parts_rest[:9]
                    matrix_string = parts_rest[9]

                    # Convert data fields
                    try:
                        data_fields = list(map(float, data_section))
                    except ValueError as e:
                        logging.error(f"{file_path}, line {line_num}: error converting data fields: {e}")
                        continue

                    # Old matrix is digits packed as groups of 3, possibly with non-digits mixed in
                    matrix_string_clean = re.sub(r"\D", "", matrix_string)
                    if len(matrix_string_clean) != expected_elems * 3:
                        logging.warning(
                            f"{file_path}, line {line_num}: matrix digit count mismatch "
                            f"(expected {expected_elems*3}, got {len(matrix_string_clean)})."
                        )
                        continue
                    matrix_cells = [int(matrix_string_clean[i:i+3])
                                    for i in range(0, len(matrix_string_clean), 3)]
                    if len(matrix_cells) != expected_elems:
                        logging.warning(f"{file_path}, line {line_num}: matrix element count mismatch.")
                        continue
                    matrix = np.array(matrix_cells, dtype=np.int32).reshape(matrix_size, matrix_size)

                    data_rows.append([timestamp, formatted_time] + data_fields)
                    if not np.all(matrix == 0):
                        matrix_data.append((timestamp, matrix))

                else:
                    # ---------------- NEW FORMAT ----------------
                    # First 19 chars must be "YYYY-MM-DD HH:MM:SS"
                    if len(line) < 20:
                        logging.warning(f"{file_path}, line {line_num}: too short for new format.")
                        continue

                    dt_str = line[:19]
                    try:
                        dt_obj = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError as e:
                        logging.error(f"{file_path}, line {line_num}: error parsing datetime '{dt_str}': {e}")
                        continue

                    timestamp = int(dt_obj.timestamp())
                    formatted_time = dt_str

                    # Rest of line (from index 20) = semicolon-separated tokens
                    rest_of_line = line[20:].strip()
                    tokens_new = [t.strip() for t in rest_of_line.split(";") if t.strip()]

                    matrix_tokens: List[str] = []
                    extra_tokens: List[str] = []
                    # Heuristic: 3-digit tokens are matrix cells; others are "extra" data
                    for token in tokens_new:
                        if re.fullmatch(r"\d{3}", token):
                            matrix_tokens.append(token)
                        else:
                            extra_tokens.append(token)

                    if len(matrix_tokens) < expected_elems:
                        # Some exporters wrap lines; try to reassemble contiguous 3-digit chunks
                        matrix_string_concat = "".join(matrix_tokens)
                        # Pad with any leftover 3-digit chunks hidden inside extra_tokens
                        matrix_string_concat += "".join([t for t in extra_tokens if re.fullmatch(r"\d{3}", t)])
                        if len(matrix_string_concat) < expected_elems * 3:
                            logging.warning(
                                f"{file_path}, line {line_num}: matrix tokens insufficient "
                                f"(have {len(matrix_string_concat)//3}, need {expected_elems})."
                            )
                            continue
                        # Trim if longer (defensive)
                        matrix_string_concat = matrix_string_concat[:expected_elems * 3]
                        matrix_cells = [int(matrix_string_concat[i:i+3])
                                        for i in range(0, len(matrix_string_concat), 3)]
                    else:
                        matrix_cells = [int(x) for x in matrix_tokens[:expected_elems]]

                    if len(matrix_cells) != expected_elems:
                        logging.warning(f"{file_path}, line {line_num}: matrix element count mismatch in new format.")
                        continue

                    matrix = np.array(matrix_cells, dtype=np.int32).reshape(matrix_size, matrix_size)

                    # Convert extra tokens to floats if possible
                    try:
                        extra_data = list(map(float, extra_tokens)) if extra_tokens else []
                    except ValueError as e:
                        logging.warning(f"{file_path}, line {line_num}: could not convert extra fields to float: {e}")
                        extra_data = []

                    data_rows.append([timestamp, formatted_time] + extra_data)
                    if not np.all(matrix == 0):
                        matrix_data.append((timestamp, matrix))

    except Exception as e:
        logging.error(f"Failed to process file '{file_path}': {e}", exc_info=True)

    return data_rows, matrix_data

# ------------------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------------------

from pathlib import Path
from typing import List, Optional, Union, Literal
import logging
import numpy as np
import pandas as pd

def save_dataframe(
    data_rows: List[List[float]],
    output_csv_path: str,
    tz: Optional[str] = None,
    safe_write: bool = True,
    append: bool = True,
    csv_style: Literal["default", "excel_es"] = "default",
    float_format: Optional[str] = None,   # e.g., "%.3f"
) -> None:
    """
    Save the extracted data rows to a CSV file, normalizing both:
      - OLD rows (9 fields): [Nº de partículas, Temp Sensor, …, Estado Sensor]
      - NEW rows (4 fields): [Intensidad máxima, Precipitación acumulada,
                              Factor de reflectividad, Estado del sensor]
    into the SAME layout so that 'Intensidad' always exists (mm/h).
    NOTE: 'Precipitacion' is kept for reference but downstream code should rely
          on 'Intensidad' only.
    """

    if not data_rows:
        logging.warning("No data rows to save.")
        return

    columns = [
        "Timestamp", "Datetime",
        "Nº de partículas", "Temp Sensor",
        "Temp cabezal derecho", "Temp cabezal izquierdo",
        "Corriente Calentador", "Intensidad",
        "Precipitacion", "Reflectividad", "Estado Sensor"
    ]

    normalized: List[List[Union[int, float, str]]] = []

    for row in data_rows:
        if len(row) < 2:
            continue
        timestamp = row[0]
        dt_str = row[1]
        rest = row[2:]

        if len(rest) == 9:
            (num_part, temp_sensor, temp_head_r, temp_head_l,
             heater_current, intensity, precipitation, reflectivity, sensor_state) = rest
        elif len(rest) == 4:
            # NEW format → map to full schema
            intensity_max, precip_acc, refl_factor, sensor_state = rest
            num_part = np.nan
            temp_sensor = np.nan
            temp_head_r = np.nan
            temp_head_l = np.nan
            heater_current = np.nan
            intensity = intensity_max          # mm/h (what we will use later)
            precipitation = precip_acc         # keep, but downstream will ignore
            reflectivity = refl_factor
        else:
            payload = list(rest) + [np.nan] * max(0, 9 - len(rest))
            (num_part, temp_sensor, temp_head_r, temp_head_l,
             heater_current, intensity, precipitation, reflectivity, sensor_state) = payload[:9]

        normalized.append([
            timestamp, dt_str, num_part, temp_sensor, temp_head_r, temp_head_l,
            heater_current, intensity, precipitation, reflectivity, sensor_state
        ])

    if not normalized:
        logging.warning("No normalized rows to save.")
        return

    df = pd.DataFrame(normalized, columns=columns)

    # --- Coerce types robustly (prevents Excel/text oddities) ---
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    num_cols = [
        "Timestamp", "Nº de partículas", "Temp Sensor", "Temp cabezal derecho",
        "Temp cabezal izquierdo", "Corriente Calentador",
        "Intensidad", "Precipitacion", "Reflectividad", "Estado Sensor"
    ]
    for c in num_cols:
        if c in df.columns:
            # accept both "." and "," decimals if upstream ever passes strings
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

    # Sort by Datetime if available; else by Timestamp
    if df["Datetime"].notna().any():
        df.sort_values(by="Datetime", inplace=True)
    else:
        df.sort_values(by="Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Write CSV ---
    output_csv_path = str(output_csv_path)

    # Style: keep defaults for pipeline files; optional ES-friendly for manual viewing
    if csv_style == "default":
        csv_kwargs = dict(sep=",", decimal=".", index=False, na_rep="")
    else:  # "excel_es"
        csv_kwargs = dict(sep=";", decimal=",", index=False, na_rep="")

    if float_format:
        csv_kwargs["float_format"] = float_format

    if append:
        file_exists = Path(output_csv_path).exists()
        chunk = 100_000
        for start in range(0, len(df), chunk):
            df.iloc[start:start+chunk].to_csv(
                output_csv_path,
                mode="a",
                header=(not file_exists and start == 0),
                **csv_kwargs
            )
    else:
        # safe write to tmp then rename
        tmp_path = f"{output_csv_path}.tmp"
        df.to_csv(tmp_path, **csv_kwargs)
        Path(tmp_path).replace(output_csv_path)

    logging.info(f"DataFrame saved to {output_csv_path} (style={csv_style})")


def save_matrices(matrix_data: List[Tuple[int, np.ndarray]],
                  output_directory: Union[str, Path],
                  subdir: str = "matrices",
                  safe_write: bool = True) -> None:
    """
    Save matrix data to .npy files.

    Parameters:
    - matrix_data: List of (timestamp, matrix).
    - output_directory: base directory where matrices subdir will be created.
    - subdir: subfolder name under output_directory (default 'matrices').
    - safe_write: use atomic writes.

    Raises:
    - Exception: if saving fails.
    """
    try:
        matrices_dir = Path(output_directory) / subdir
        ensure_directory_exists(str(matrices_dir))
        logging.info(f"Matrices will be saved to {matrices_dir}")

        # Sort by timestamp for deterministic order
        for timestamp, matrix in sorted(matrix_data, key=lambda x: x[0]):
            matrix_path = matrices_dir / f"matrix_{timestamp}.npy"
            _safe_write_npy(np.asarray(matrix, dtype=np.int32, order="C"), matrix_path, safe=safe_write)
            logging.debug(f"Matrix saved to {matrix_path}")

        logging.info(f"All matrices have been saved to {matrices_dir}")
    except Exception as e:
        logging.error(f"Failed to save matrices to {output_directory}: {e}", exc_info=True)
        raise

# ------------------------------------------------------------------------------
# Missing timestamp audit 
# ------------------------------------------------------------------------------


def infer_sampling_minutes(dt_series: pd.Series) -> int:
    """Infer sampling step (minutes) from a datetime series (median Δt)."""
    dt = pd.to_datetime(dt_series, errors="coerce").dropna()
    if dt.size < 2:
        return 1
    deltas = dt.diff().dropna().dt.total_seconds() / 60.0
    step_min = max(1, int(round(float(deltas.median()))))
    return step_min

def find_missing_timestamps(df: pd.DataFrame,
                            freq_min: Optional[int] = None,
                            tz: Optional[str] = None) -> Dict[str, Any]:
    """
    Return missing timestamps and outage groups for a processed DF that has 'Datetime'.
    Handles tz-naive/aware mixes and converts consistently when `tz` is provided.
    """
    if "Datetime" not in df.columns:
        raise ValueError("DataFrame must contain 'Datetime' column.")

    # Coerce to datetimes
    dt = pd.to_datetime(df["Datetime"], errors="coerce").dropna()
    if dt.empty:
        return {
            "freq_min": int(freq_min or 1),
            "expected": 0, "actual": 0, "missing": 0, "missing_pct": 0.0,
            "window": (None, None), "missing_list": [], "outages": []
        }

    # Normalize timezone safely
    if tz:
        # If series is tz-naive, localize; if tz-aware, convert
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(tz)
        else:
            dt = dt.dt.tz_convert(tz)

    # Infer sampling step if not provided
    if freq_min is None:
        deltas_min = dt.diff().dropna().dt.total_seconds() / 60.0
        freq = max(1, int(round(float(deltas_min.median())))) if len(deltas_min) else 1
    else:
        freq = int(freq_min)

    # Build expected vs. actual ranges
    start, end = dt.min(), dt.max()
    expected = pd.date_range(start, end, freq=f"{freq}T", tz=getattr(start, "tz", None))
    actual = pd.DatetimeIndex(dt.unique())

    missing_idx = expected.difference(actual)

    # Group consecutive missing stamps into outages
    outages: List[Dict[str, Any]] = []
    if len(missing_idx) > 0:
        s = pd.Series(missing_idx)
        grp = (s.diff() != pd.Timedelta(minutes=freq)).cumsum()
        for _, g in s.groupby(grp):
            outages.append({
                "start": g.iloc[0],
                "end": g.iloc[-1],
                "count": int(len(g)),
                "duration_min": int(len(g) * freq),
            })

    return {
        "freq_min": int(freq),
        "expected": int(len(expected)),
        "actual": int(len(actual)),
        "missing": int(len(missing_idx)),
        "missing_pct": (100.0 * len(missing_idx) / max(1, len(expected))),
        "window": (start, end),
        "missing_list": [ts for ts in missing_idx],
        "outages": outages,
    }

def write_missing_report(summary: Dict[str, Any],
                         site: str,
                         logs_dir: Union[str, Path],
                         write_csv: bool = True) -> Dict[str, Path]:
    """
    Write a human-readable TXT + optional CSV of missing timestamps.
    Returns paths of written files.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    txt_path = logs_dir / f"missing_timestamps_{site}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Site: {site}\n")
        f.write(f"Window: {summary['window'][0]} — {summary['window'][1]}\n")
        f.write(f"Sampling (min): {summary['freq_min']}\n")
        f.write(f"Expected: {summary['expected']}, Actual: {summary['actual']}, "
                f"Missing: {summary['missing']} ({summary['missing_pct']:.2f}%)\n\n")
        f.write("Outages:\n")
        if summary["outages"]:
            for o in summary["outages"]:
                f.write(f"- {o['start']} → {o['end']}  "
                        f"({o['count']} stamps, {o['duration_min']} min)\n")
        else:
            f.write("- None\n")

    paths = {"txt": txt_path}

    if write_csv:
        import pandas as pd  # local import
        csv_path = logs_dir / f"missing_timestamps_{site}.csv"
        pd.Series(summary["missing_list"], name="missing_datetime").to_csv(csv_path, index=False)
        paths["csv"] = csv_path

    logging.info(f"[{site}] Missing timestamps report written: {paths}")
    return paths
