import os
import csv
import shutil
import tempfile
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any

@contextmanager
def atomic_write(filepath: str, mode: str = "w", encoding: str = "utf-8", **kwargs):
    """
    Context manager for atomic file writing.
    Writes to a temp file first, then moves it to destination on success.
    """
    filepath = Path(filepath)
    parent = filepath.parent
    parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in the same directory to ensure atomic move
    fd, temp_path = tempfile.mkstemp(dir=parent, text="b" not in mode)
    os.close(fd)
    
    try:
        with open(temp_path, mode, encoding=encoding, **kwargs) as f:
            yield f
        # Move temp to final
        shutil.move(temp_path, filepath)
    except Exception as e:
        # Cleanup
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise e

def safe_write_csv(filepath: str, data: List[Dict[str, Any]], fieldnames: List[str]):
    """Write list of dicts to CSV atomically."""
    if not data:
        logging.warning(f"No data to write to {filepath}")
        return

    try:
        with atomic_write(filepath, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logging.info(f"Saved CSV: {filepath}")
    except Exception as e:
        logging.error(f"Failed to write CSV {filepath}: {e}")
        raise

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
