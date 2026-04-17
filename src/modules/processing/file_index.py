import logging
from pathlib import Path
from typing import Dict, List, Optional
import json

from ..models import FormatType
from ..io_utils import atomic_write

class FileIndex:
    """
    Index of available DSD files (Raw and Matrices).
    Used to deterministically find inputs for a given Event time window.
    """
    
    def __init__(self, raw_dir: Path, processed_dir: Path, site: str):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.site = site
        self.matrices_dir = processed_dir / site / "matrices"
        
        # Cache
        self.matrix_map: Dict[int, Path] = {} # timestamp -> path
        
    def build_matrix_index(self):
        """Scan processed matrices directory."""
        if not self.matrices_dir.exists():
            return
            
        count = 0
        for f in self.matrices_dir.glob("matrix_*.npy"):
            try:
                # Filename: matrix_1735866000.npy
                ts_part = f.stem.split("_")[1]
                ts = int(ts_part)
                self.matrix_map[ts] = f
                count += 1
            except:
                pass
        logging.info(f"Indexed {count} matrices for {self.site}")
                
    def get_matrices_for_window(self, start_ts: int, end_ts: int) -> List[Path]:
        """Return list of matrix paths within window [start, end]. Sorted."""
        # Naive scan of dict keys? 
        # If we have 1M keys, better to use sorted list + bisect.
        # For now (weeks/months of data), naiive filter is OK (~50k keys).
        
        found = []
        # Optimization: Sort keys once?
        for ts, path in self.matrix_map.items():
            if start_ts <= ts <= end_ts:
                found.append((ts, path))
                
        found.sort(key=lambda x: x[0])
        return [p for t, p in found]
