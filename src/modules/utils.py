# utils.py
"""
Utility functions for the DSD app:
  - Directory management
  - Logging configuration
  - Config loading with sane defaults
  - Filename date parsing
  - Config-driven cleanup (pattern-based, dry-run, keep rules)
"""

from __future__ import annotations

import os, fnmatch
import re
import sys
import yaml
import shutil
import logging
import fnmatch
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple, Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path



# ------------------------------------------------------------------------------
# Paths / directories
# ------------------------------------------------------------------------------

def get_base_path() -> Path:
    """
    Returns the project base path. If packaged (frozen), resolves relative to the executable;
    otherwise relative to this file (two levels up: src/modules/utils.py -> project root).
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # PyInstaller-style
        return Path(sys.executable).resolve().parent
    # Assume standard source layout: <project>/src/modules/utils.py
    return Path(__file__).resolve().parents[2]


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Create a directory (and parents) if it doesn't exist. Returns the resolved Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

def setup_initial_logging(level: int = logging.INFO) -> None:
    """
    Simple console logging, useful very early in startup.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s:%(levelname)s:%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def configure_logging(
    log_file_path: Union[str, Path],
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    log_level: int = logging.INFO
) -> None:
    """
    Configure logging with a rotating file handler + console handler.
    """
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(log_level)

    # Ensure log directory
    log_file_path = Path(log_file_path)
    ensure_directory_exists(log_file_path.parent)

    # File (rotating)
    file_handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))

    # Console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


# ------------------------------------------------------------------------------
# Config loading (with robust defaults for new keys)
# ------------------------------------------------------------------------------

def load_config(config_file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load YAML config and provide defaults for new blocks (paths/run/events/visualization/cleanup).
    Keeps existing top-level start_date/end_date for GUI compatibility.
    """
    base_path = get_base_path()
    config_path = Path(config_file_path) if config_file_path else (base_path / "config" / "config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    # Defaults so the rest of the code can rely on keys existing
    cfg.setdefault("paths", {})
    paths = cfg["paths"]
    paths.setdefault("base_dir", str(base_path))
    paths.setdefault("raw_dir", "data/raw")
    paths.setdefault("processed_dir", "data/processed")
    paths.setdefault("events_dir", "data/processed/events")
    paths.setdefault("plots_dir", "plots")
    paths.setdefault("logs_dir", "logs")
    paths.setdefault("diam_vel_mapping", "diam_vel_mapping.csv")

    cfg.setdefault("run", {})
    cfg["run"].setdefault("timezone", cfg.get("timezone", "America/Argentina/Cordoba"))
    cfg["run"].setdefault("sites", ["bosque_alegre", "villa_dique", "pilar"])

    cfg.setdefault("processing", {})
    cfg.setdefault("events", {
        "intensity_threshold_mm_h": 0.1,
        "min_duration_min": 10,
        "min_gap_min": 30,
        "min_accum_mm": 0.2,
        "merge_if_gap_min": 5,
        "outputs": {
            "save_per_event_combined": True,
            "save_period_combined": True,
        },
    })
    cfg.setdefault("visualization", {
        "hyetograph": {"bin_minutes": 10},
        "export": {"formats": ["png"], "dpi": 150, "tight_layout": True, "per_event_subdirs": True},
    })
    cfg.setdefault("io", {
        "filename_patterns": {
            "event_csv": "event_{id}.csv",
            "combined_npy": "combined_event_{id}.npy",
            "combined_period_npy": "combined_period_{start}_{end}.npy",
        },
        "safe_writes": True,
    })
    cfg.setdefault("logging", {"level": "INFO", "rotation": {"max_mb": 10, "backup_count": 5}})
    cfg.setdefault("cleanup", {
        "enabled": False,
        "when": "always",
        "dry_run": True,
        "keep_patterns": ["**/.gitkeep", "**/README.md"],
        "targets": [],
    })

    return cfg


# ------------------------------------------------------------------------------
# Filename → datetime parsing
# ------------------------------------------------------------------------------

# Common patterns: YYYYMMDD[_-]HHMMSS, YYYY-MM-DD, YYYYMMDD, etc.
_DATE_PATTERNS = [
    # 2024-12-31_235959 or 2024-12-31-235959
    re.compile(r"(?P<y>\d{4})[-_](?P<m>\d{2})[-_](?P<d>\d{2})[-_](?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"),
    # 20241231_235959 or 20241231-235959
    re.compile(r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})[-_](?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"),
    # 2024-12-31
    re.compile(r"(?P<y>\d{4})[-_](?P<m>\d{2})[-_](?P<d>\d{2})"),
    # 20241231
    re.compile(r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})"),
]

def parse_filename_to_date(name: Union[str, Path]) -> Optional[datetime]:
    """
    Try to parse a datetime from a filename. Returns None if nothing matches.
    Missing HHMMSS default to 00:00:00.
    """
    s = Path(name).name
    for pat in _DATE_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        parts = m.groupdict()
        y = int(parts["y"]); mth = int(parts["m"]); d = int(parts["d"])
        H = int(parts.get("H") or 0); M = int(parts.get("M") or 0); S = int(parts.get("S") or 0)
        try:
            return datetime(y, mth, d, H, M, S)
        except ValueError:
            continue
    return None


def coerce_date_str(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def iter_files_with_ext(root: Path, allowed_exts=(".txt", ".dat")):
    if not Path(root).exists():
        return []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed_exts:
            yield p

def filter_files_by_window(files, start_dt: datetime = None, end_dt: datetime = None):
    out = []
    for f in files:
        dt = parse_filename_to_date(f.name)
        if dt is None:
            out.append(f)  # keep if no date in name
            continue
        if (start_dt is None or dt >= start_dt) and (end_dt is None or dt <= end_dt):
            out.append(f)
    return sorted(out)


# ------------------------------------------------------------------------------
# Cleanup (NEW): pattern-based with dry-run and keep rules
# ------------------------------------------------------------------------------

def _is_under(base: Path, candidate: Path) -> bool:
    """
    Return True if 'candidate' is the same as 'base' or located under 'base'.
    Works on Python 3.8 (no Path.is_relative_to required).
    """
    base = base.resolve()
    cand = candidate.resolve()
    return cand == base or str(cand).startswith(str(base) + os.sep)


def cleanup_from_config(cfg: Dict[str, Any], date_changed: bool = True) -> None:
    """
    New mode: pattern-based cleanup with dry-run, keep patterns, and safety fences.

    Expects config.cleanup like:
      enabled: true
      when: "always" | "date_change" | "never"
      dry_run: true|false
      keep_patterns: ["**/.gitkeep", "**/README.md"]
      targets:
        - name: plots
          path: "plots"
          patterns: ["**/*"]
        - name: matrices
          paths: ["data/processed/site1/matrices", ...]
          patterns: ["**/*"]
    """
    c = (cfg or {}).get("cleanup", {})
    if not c.get("enabled", False):
        logging.info("Cleanup: disabled")
        return

    when = c.get("when", "always")
    if when == "never" or (when == "date_change" and not date_changed):
        logging.info("Cleanup: skipped due to 'when' policy")
        return

    dry = bool(c.get("dry_run", False))
    keep_patterns = list(c.get("keep_patterns", []))

    # Base directory to fence deletions. Prefer config.paths.base_dir if present.
    base_dir = Path((cfg.get("paths", {}) or {}).get("base_dir", get_base_path())).resolve()

    def should_keep(p: Path) -> bool:
        sp = str(p)
        return any(fnmatch.fnmatch(sp, pat) for pat in keep_patterns)

    def wipe(path: Path, patterns: List[str]) -> None:
        if not path.exists():
            logging.info(f"Cleanup: path not found, skipping -> {path}")
            return
        for pat in (patterns or ["**/*"]):
            for target in path.glob(pat):
                # Safety fences
                if not _is_under(base_dir, target):
                    logging.warning(f"Cleanup: refused (outside base) -> {target}")
                    continue
                if should_keep(target):
                    continue
                if dry:
                    logging.info(f"[DRY-RUN] would remove: {target}")
                else:
                    if target.is_dir():
                        shutil.rmtree(target, ignore_errors=True)
                        logging.info(f"Removed dir: {target}")
                    else:
                        try:
                            target.unlink()
                            logging.info(f"Removed file: {target}")
                        except FileNotFoundError:
                            pass

    targets = c.get("targets", [])
    for t in targets:
        # Support a single 'path' or a list in 'paths'
        raw_paths = []
        if "path" in t:
            raw_paths.append(t["path"])
        if "paths" in t:
            raw_paths.extend(t["paths"])
        patterns = t.get("patterns", ["**/*"])
        for p in raw_paths:
            wipe(Path(os.path.expandvars(p)), patterns)


# Backward-compatible alias: existing call sites can keep calling cleanup_output
def cleanup_output(config: Dict[str, Any], date_changed: bool = True) -> None:
    """
    Compatibility alias that routes to the new cleanup implementation.
    """
    cleanup_from_config(config, date_changed=date_changed)


def dedup_and_sort_output_csv(csv_path: Path,
                              subset=('Timestamp','Datetime'),
                              keep='last') -> Tuple[int, int]:
    """
    Deduplicate and sort output_data.csv atomically.
    Returns (rows_before, rows_after).
    """
    p = Path(csv_path).resolve()
    if not p.exists():
        logging.warning(f"[dedup] File not found, skipping: {p}")
        return (0, 0)

    df = pd.read_csv(p)
    before = len(df)

    # normalize types so duplicates match
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce').astype('Int64')

    sort_cols = [c for c in ['Timestamp','Datetime'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    subset_cols = [c for c in subset if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols, keep=keep)
    else:
        df = df.drop_duplicates(keep=keep)

    after = len(df)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)
    logging.info(f"[dedup] {p.name}: {before} → {after} rows (removed {before-after}).")
    return (before, after)

def atomic_save_npy(array: np.ndarray, out_path: Path) -> None:
    """
    Save a .npy atomically to avoid partial files or race-created *.tmp.npy.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(out_path.parent), delete=False, suffix=".tmp.npy") as tmp:
        np.save(tmp.name, array)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, str(out_path))

    
def find_files(
    root_directory: Path,
    file_patterns: Iterable[str] = ("*.txt",),
    exclude_dirs: Optional[Iterable[Path]] = None,
    exclude_files: Optional[Iterable[str]] = None,
    exclude_file_patterns: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Walks root_directory and returns paths matching any of file_patterns.
    Excludes any path under exclude_dirs, filenames in exclude_files,
    and names matching exclude_file_patterns (glob/fnmatch patterns).
    """
    root_directory = Path(root_directory).resolve()
    ex_dirs = {Path(p).resolve() for p in (exclude_dirs or [])}
    ex_files = set(exclude_files or [])
    ex_file_pats = list(exclude_file_patterns or [])

    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # prune excluded directories
        abs_dir = Path(dirpath).resolve()
        # If this directory is within any excluded dir, skip walking it
        if any(str(abs_dir).startswith(str(ed) + os.sep) or abs_dir == ed for ed in ex_dirs):
            dirnames[:] = []
            continue

        # filter files
        for name in filenames:
            # filename-level excludes
            if name in ex_files:
                continue
            if any(fnmatch.fnmatch(name, pat) for pat in ex_file_pats):
                continue
            # positive patterns
            if any(fnmatch.fnmatch(name, pat) for pat in file_patterns):
                out.append(str(Path(dirpath) / name))
    return out