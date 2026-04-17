import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any

DEFAULTS = {
    "parsing": {
        "timezone": "America/Argentina/Cordoba",
        "detect_lines": 3,
        "stop_on_unknown_format": True
    },
    "formats": {
        "dsd2": {"colon_fields": 9, "packed_digits_per_cell": 3},
        "vst": {"matrix_cells": 1024, "trailing_fields": 4},
        "dsd1": {"header_fields": 7, "matrix_cells": 1024}
    },
    "events": {
        "intensity_threshold": 0.1,
        "inter_event_minutes": 30,
        "min_event_depth_mm": 0.1
    },
    "parameterize": {
        "enabled": True,
        "stop_on_event_error": False,
        "output_dir": "data/processed/event_parameters"
    },
    "paths": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "events_dir": "data/processed/events"
    }
}

def load_config(path: str) -> Dict[str, Any]:
    
    path_obj = Path(path)
    if not path_obj.exists():
        logging.warning(f"Config file {path} not found. Using defaults.")
        return DEFAULTS.copy()
    
    try:
        with open(path_obj, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        logging.error(f"Error parseando config {path}: {e}")
        return DEFAULTS.copy()
        
    return _apply_defaults(user_cfg, DEFAULTS)

def _apply_defaults(user: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge user config with defaults."""
    # Start with a copy of defaults
    merged = default.copy()
    
    for k, v in user.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _apply_defaults(v, merged[k])
        else:
            merged[k] = v
            
    return merged

def validate_config(cfg: Dict[str, Any]) -> bool:
    """Basic validation logic."""
    # Todo: Add schema validation
    return True
