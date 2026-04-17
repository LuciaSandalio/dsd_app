from dataclasses import dataclass, field
from typing import List, Optional, Any
from enum import Enum
import numpy as np

class FormatType(Enum):
    UNKNOWN = "unknown"
    DSD2_PACKED = "dsd2_packed"     # Colon + Packed Blob (Old)
    VST_SEMICOLON = "vst_semicolon" # DateTime + Colon + Semicolons (New)
    DSD1_SEMICOLON = "dsd1_semicolon" # Epoch + Semicolons (Hybrid?)

@dataclass
class DetectedFormat:
    fmt: FormatType
    confidence: float
    details: str = ""

@dataclass
class SampleRecord:
    """Canonical representation of a single DSD record."""
    timestamp: int          # Epoch seconds (UTC/Reference)
    datetime_str: str       # ISO formatted string
    intensity: float        # mm/h
    manager_intensity: float # from header if available, else same as intensity
    
    # Core fields
    precip_accum: Optional[float] = None
    reflectivity: Optional[float] = None
    status: int = 0
    
    # Optional sensors
    particles_count: Optional[int] = None
    temp_housing: Optional[float] = None
    temp_head_r: Optional[float] = None
    temp_head_l: Optional[float] = None
    heater_current: Optional[float] = None
    
    # Matrix (stored separately in file, but kept here during processing if needed)
    # 32x32 array. Flattened or object? 
    # For memory efficiency in large merges, we might NOT store matrix here 
    # but write it immediately. But for the parser interface, returning it is fine.
    matrix: Optional[np.ndarray] = field(default=None, repr=False)
    
    # Raw extras
    spectrum_nd: Optional[List[float]] = field(default=None, repr=False)
    velocity_vd: Optional[List[float]] = field(default=None, repr=False)

@dataclass
class ParseStats:
    file_path: str
    format: str
    total_lines: int = 0
    parsed_lines: int = 0
    skipped_lines: int = 0
    matrix_count: int = 0
    errors: dict = field(default_factory=dict) # e.g. {"ValueError": 5}

@dataclass
class IngestResult:
    site: str
    files_processed: int
    total_rows: int
    matrices_saved: int
    output_csv: str
    manifest_path: str
