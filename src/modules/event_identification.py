#!/usr/bin/env python3
# src/modules/event_identification.py
# Event labeling utilities, diagnostics, and side-effect helpers.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from modules.utils import ensure_directory_exists

# ---------------------------------------------------------------------------
# I/O helpers (atomic CSV)
# ---------------------------------------------------------------------------

def _safe_write_csv(df: pd.DataFrame, dest: Path, safe: bool = True) -> None:
    dest = Path(dest)
    ensure_directory_exists(dest.parent)
    if safe:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(dest)
    else:
        df.to_csv(dest, index=False)

# ---------------------------------------------------------------------------
# Core helpers (rate → depth, dedup, timestamp handling)
# ---------------------------------------------------------------------------

def integrate_rate_to_depth_mm(
    df: pd.DataFrame,
    *,
    time_col: str = "Datetime",
    rate_col: str = "Intensidad",
    cap_gap_minutes: Optional[int] = 30,
    fill_first: str = "median",  # "median" or "zero"
) -> pd.Series:
    """
    Convert rainfall rate (mm/h) into per-sample depth (mm) using Δt between rows.

    - df[time_col]: timestamp column (string or datetime-like)
    - df[rate_col]: instantaneous rain rate in mm/h
    - cap_gap_minutes: cap Δt to avoid over-integration across outages (None to disable)
    - fill_first: how to fill Δt of the first row ("median" positive Δt of series, or "zero")

    Returns a Series aligned to df.index named "Depth_mm".
    """
    if time_col not in df.columns or rate_col not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index, dtype=float, name="Depth_mm")

    t = pd.to_datetime(df[time_col], errors="coerce")
    r = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0)

    mask = t.notna() & r.notna()
    if not mask.any():
        return pd.Series([0.0] * len(df), index=df.index, dtype=float, name="Depth_mm")

    t_valid = t[mask]
    dt_s = t_valid.diff().dt.total_seconds()

    if fill_first == "median":
        med = dt_s[dt_s > 0].median()
        if not pd.notna(med):
            med = 60.0
        dt_s = dt_s.fillna(med)
    else:
        dt_s = dt_s.fillna(0.0)

    if cap_gap_minutes is not None and cap_gap_minutes > 0:
        cap_s = float(cap_gap_minutes) * 60.0
        dt_s = dt_s.clip(upper=cap_s)

    depth_mm_valid = r[mask].values * (dt_s.values / 3600.0)

    out = pd.Series(0.0, index=df.index, dtype=float, name="Depth_mm")
    out.loc[mask] = depth_mm_valid
    return out

def remove_duplicates(df: pd.DataFrame, subset_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows based on subset columns (default: Timestamp/Datetime)."""
    if subset_columns is None:
        subset_columns = [c for c in ("Timestamp", "Datetime") if c in df.columns] or df.columns.tolist()
    before = len(df)
    dedup = df.drop_duplicates(subset=subset_columns, keep="last").copy()
    return dedup, before - len(dedup)

def ensure_continuous_timestamps(df: pd.DataFrame, freq: Optional[str] = None, method: Optional[str] = None) -> pd.DataFrame:
    """Reindex to a regular time grid on 'Datetime'. If freq None, infer from median step."""
    if "Datetime" not in df.columns:
        return df
    work = df.copy()
    work["Datetime"] = pd.to_datetime(work["Datetime"], errors="coerce")
    work = work.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    if freq is None:
        deltas = work["Datetime"].diff().dropna().dt.total_seconds() / 60.0
        if len(deltas) == 0:
            return work
        step_min = max(1, int(round(np.median(deltas))))
        freq = f"{step_min}T"

    idx = pd.date_range(work["Datetime"].min(), work["Datetime"].max(), freq=freq, tz=work["Datetime"].dt.tz)
    work = work.set_index("Datetime").reindex(idx)
    if method in ("ffill", "bfill"):
        work = getattr(work, method)()
    return work.reset_index().rename(columns={"index": "Datetime"})

def _compute_accum_from_rate(df: pd.DataFrame, *, max_gap_min: Optional[float] = None) -> pd.Series:
    """Backed by integrate_rate_to_depth_mm for a single source of truth."""
    cap = None if max_gap_min in (None, "", float("nan")) else int(max_gap_min)
    return integrate_rate_to_depth_mm(
        df,
        time_col="Datetime",
        rate_col="Intensidad",
        cap_gap_minutes=cap,
        fill_first="median",
    )

def _build_is_raining_hysteresis(
    rate_mm_h: pd.Series,
    *,
    start_thr: float,
    stop_thr: float,
    min_wet_streak: int = 1,
    min_dry_streak: int = 1,
) -> np.ndarray:
    """
    Return boolean array using hysteresis + min consecutive wet/dry streaks.
    start when rate >= start_thr and (wet_streak >= min_wet_streak),
    stop  when rate <  stop_thr and (dry_streak >= min_dry_streak).
    """
    r = pd.to_numeric(rate_mm_h, errors="coerce").fillna(0.0).to_numpy()
    n = r.shape[0]
    out = np.zeros(n, dtype=bool)
    state = False
    wet_streak = 0
    dry_streak = 0
    for i in range(n):
        if state:
            if r[i] < stop_thr:
                dry_streak += 1
                wet_streak = 0
                if dry_streak >= min_dry_streak:
                    state = False
                    dry_streak = 0
            else:
                dry_streak = 0
                wet_streak += 1
        else:
            if r[i] >= start_thr:
                wet_streak += 1
                dry_streak = 0
                if wet_streak >= min_wet_streak:
                    state = True
                    wet_streak = 0
            else:
                wet_streak = 0
                dry_streak += 1
        out[i] = state
    return out

# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

@dataclass
class Event:
    id: int
    site: str
    start: pd.Timestamp
    end: pd.Timestamp
    duration_min: float
    accum_mm: float
    max_intensity_mm_h: float
    mean_intensity_mm_h: float
    median_intensity_mm_h: float
    p95_intensity_mm_h: float
    max_gap_inside_min: float
    n_points: int

# ---------------------------------------------------------------------------
# Core identification
# ---------------------------------------------------------------------------

def mark_precipitation_activity(df: pd.DataFrame, intensity_threshold_mm_h: float) -> pd.DataFrame:
    """Simple threshold mark (kept for API stability)."""
    work = df.copy()
    work["Intensidad"] = pd.to_numeric(work.get("Intensidad", pd.Series(index=work.index)), errors="coerce")
    work["is_raining"] = (work["Intensidad"].fillna(0) >= float(intensity_threshold_mm_h))
    return work

def identify_precipitation_events(
    df: pd.DataFrame,
    *,
    intensity_threshold_mm_h: float,
    min_duration_min: int,
    min_gap_min: int,
    min_accum_mm: float,
    merge_if_gap_min: Optional[int] = None,
    # hysteresis knobs
    start_threshold_mm_h: Optional[float] = None,
    stop_threshold_mm_h: Optional[float] = None,
    min_wet_streak: int = 1,
    min_dry_streak: int = 1,
    max_gap_for_rate_integration_min: Optional[float] = None,
    require_both_filters: bool = True,
    site: str = "",
) -> Tuple[pd.DataFrame, List[Event]]:
    """
    Segment events with hysteresis (if provided), gap rules, and filters.
    Returns annotated DF + list of Event dataclasses (deterministic IDs by start time).
    """
    if "Datetime" not in df.columns:
        raise ValueError("DataFrame must have 'Datetime' column.")
    if "Intensidad" not in df.columns:
        raise ValueError("DataFrame must have 'Intensidad' (mm/h) column.")

    work = df.copy()
    work["Datetime"] = pd.to_datetime(work["Datetime"], errors="coerce")
    work["Intensidad"] = pd.to_numeric(work["Intensidad"], errors="coerce")
    work = work.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    # Accumulation per row from instantaneous rate
    work["accum_from_rate_mm"] = _compute_accum_from_rate(work, max_gap_min=max_gap_for_rate_integration_min)

    # Wet/dry via hysteresis (preferred) or simple threshold
    if start_threshold_mm_h is not None and stop_threshold_mm_h is not None:
        wet = _build_is_raining_hysteresis(
            work["Intensidad"],
            start_thr=float(start_threshold_mm_h),
            stop_thr=float(stop_threshold_mm_h),
            min_wet_streak=int(min_wet_streak),
            min_dry_streak=int(min_dry_streak),
        )
    else:
        wet = (work["Intensidad"].fillna(0) >= float(intensity_threshold_mm_h)).to_numpy()

    work["is_raining"] = wet
    times = work["Datetime"].to_numpy()

    # Preliminary segmentation, considering min_gap_min
    event_id = np.full(len(work), -1, dtype=int)
    current_id = -1
    for i in range(len(work)):
        if wet[i]:
            start_new = (i == 0) or (not wet[i - 1])
            if not start_new:
                delta = times[i] - times[i - 1]
                try:
                    gap_min = float(delta / np.timedelta64(1, "m"))
                except Exception:
                    gap_min = (delta.total_seconds() / 60.0)
                if gap_min >= float(min_gap_min):
                    start_new = True
            if start_new:
                current_id += 1
            event_id[i] = current_id
    work["event_id"] = event_id

    # Optionally merge short dry-gapped neighbors
    if merge_if_gap_min and merge_if_gap_min > 0:
        spans = []
        for eid, grp in work[work["event_id"] >= 0].groupby("event_id"):
            spans.append((eid, grp["Datetime"].min(), grp["Datetime"].max()))
        spans.sort(key=lambda t: t[1])
        merges: Dict[int, int] = {}
        for (eid1, s1, e1), (eid2, s2, e2) in zip(spans, spans[1:]):
            gap_min = (s2 - e1).total_seconds() / 60.0
            if gap_min <= int(merge_if_gap_min):
                merges[eid2] = eid1
        if merges:
            def root(x):
                while x in merges:
                    x = merges[x]
                return x
            remapped = np.array([root(e) if e >= 0 else e for e in work["event_id"]], dtype=int)
            valid = sorted({e for e in remapped if e >= 0})
            id_map = {old: new for new, old in enumerate(valid)}
            work["event_id"] = np.array([id_map.get(e, -1) for e in remapped], dtype=int)

    # Compute diagnostics, apply filters, finalize IDs
    events: List[Event] = []
    out_ids = [e for e in sorted(work["event_id"].unique()) if e >= 0]
    final_id_map: Dict[int, int] = {}
    next_id = 0

    for eid in out_ids:
        block = work[work["event_id"] == eid]
        if block.empty:
            continue
        start = block["Datetime"].iloc[0]
        end = block["Datetime"].iloc[-1]
        duration_min = (end - start).total_seconds() / 60.0
        accum_mm = float(block["accum_from_rate_mm"].sum())
        intens = block["Intensidad"].astype(float)
        max_int = float(np.nanmax(intens)) if len(intens) else float("nan")
        mean_int = float(np.nanmean(intens)) if len(intens) else float("nan")
        median_int = float(np.nanmedian(intens)) if len(intens) else float("nan")
        p95_int = float(np.nanpercentile(intens, 95)) if len(intens) else float("nan")
        gaps_inside = block["Datetime"].diff().dt.total_seconds().div(60.0).fillna(0.0)
        max_gap_inside = float(np.nanmax(gaps_inside)) if len(gaps_inside) else 0.0

        pass_duration = duration_min >= float(min_duration_min)
        pass_accum = accum_mm >= float(min_accum_mm)
        keep = (pass_duration and pass_accum) if require_both_filters else (pass_duration or pass_accum)
        if not keep:
            continue

        final_id_map[eid] = next_id
        events.append(Event(
            id=next_id,
            site=site,
            start=start,
            end=end,
            duration_min=duration_min,
            accum_mm=accum_mm,
            max_intensity_mm_h=max_int,
            mean_intensity_mm_h=mean_int,
            median_intensity_mm_h=median_int,
            p95_intensity_mm_h=p95_int,
            max_gap_inside_min=max_gap_inside,
            n_points=len(block),
        ))
        next_id += 1

    work["event_id"] = work["event_id"].map(lambda x: final_id_map.get(x, -1)) if final_id_map else -1
    return work, events

# ---------------------------------------------------------------------------
# Saving annotated data and per-event extracts
# ---------------------------------------------------------------------------

def save_annotated_data(df_annotated: pd.DataFrame, output_dir: Path, safe_write: bool = True) -> Path:
    out = Path(output_dir) / "annotated_data.csv"
    _safe_write_csv(df_annotated, out, safe=safe_write)
    logging.info(f"Annotated data saved: {out}")
    return out

def extract_and_save_events(
    df_annotated: pd.DataFrame,
    events: List[Event],
    output_dir: Path,
    filename_pattern: str = "event_{id}.csv",
    safe_write: bool = True
) -> List[Path]:
    ensure_directory_exists(output_dir)
    written: List[Path] = []
    for ev in events:
        ev_df = df_annotated[df_annotated["event_id"] == ev.id].copy()
        out = Path(output_dir) / filename_pattern.format(id=ev.id)
        _safe_write_csv(ev_df, out, safe=safe_write)
        written.append(out)
        logging.info(f"Event {ev.id} saved: {out}")
    return written

# ---------------------------------------------------------------------------
# Combined matrices per event / period
# ---------------------------------------------------------------------------

def _iter_matrix_files_in_range(matrices_dir: Path, t_start: int, t_end: int) -> List[Path]:
    """Return matrix_*.npy files whose <timestamp> is in [t_start, t_end]."""
    found: List[Tuple[int, Path]] = []
    for p in Path(matrices_dir).glob("matrix_*.npy"):
        try:
            ts = int(p.stem.split("_", 1)[1])
        except Exception:
            continue
        if t_start <= ts <= t_end:
            found.append((ts, p))
    return [p for _, p in sorted(found, key=lambda x: x[0])]

def _combine_matrices(files: List[Path], matrix_size: int, mode: str = "sum", scale: float = 1.0) -> np.ndarray:
    """
    Combine matrices with a mode:
      - "sum": plain sum
      - "mean_per_sample": sum / N
      - "normalize_to_total_mm": sum / scale   (scale = event accumulation in mm)
    """
    if not files:
        return np.zeros((matrix_size, matrix_size), dtype=np.float32)
    acc = None
    n = 0
    for f in files:
        try:
            arr = np.load(f, mmap_mode="r")
            if arr.shape != (matrix_size, matrix_size):
                logging.warning(f"Matrix shape mismatch in {f}: {arr.shape}")
                continue
            arr = np.asarray(arr, dtype=np.float32)
            acc = arr if acc is None else (acc + arr)
            n += 1
        except Exception as e:
            logging.warning(f"Failed to load matrix {f}: {e}")
    if acc is None:
        acc = np.zeros((matrix_size, matrix_size), dtype=np.float32)

    mode = (mode or "sum").lower()
    if mode == "mean_per_sample" and n > 0:
        acc = acc / float(n)
    elif mode == "normalize_to_total_mm" and scale and np.isfinite(scale) and scale > 0:
        acc = acc / float(scale)
    return acc

def save_combined_event_matrix(
    matrices_dir: Path,
    event: Event,
    out_path: Path,
    matrix_size: int = 32,
    safe_write: bool = True,
    mode: str = "sum",
) -> Optional[Path]:
    t_start = int(event.start.timestamp())
    t_end = int(event.end.timestamp())
    files = _iter_matrix_files_in_range(matrices_dir, t_start, t_end)
    combined = _combine_matrices(files, matrix_size=matrix_size, mode=mode, scale=event.accum_mm)

    ensure_directory_exists(out_path.parent)
    if safe_write:
        with NamedTemporaryFile(dir=out_path.parent, delete=False, suffix=out_path.suffix) as tf:
            np.save(tf, combined)  # ends with .npy
            tmp = Path(tf.name)
        tmp.replace(out_path)
    else:
        np.save(out_path, combined)

    logging.info(f"Combined matrix saved for event {event.id}: {out_path} (files={len(files)})")
    return out_path

def save_combined_period_matrix(
    matrices_dir: Path,
    start_ts: int,
    end_ts: int,
    out_path: Path,
    matrix_size: int = 32,
    safe_write: bool = True,
    mode: str = "sum",
) -> Optional[Path]:
    files = _iter_matrix_files_in_range(matrices_dir, start_ts, end_ts)
    combined = _combine_matrices(files, matrix_size=matrix_size, mode=mode, scale=1.0)

    ensure_directory_exists(out_path.parent)
    if safe_write:
        with NamedTemporaryFile(dir=out_path.parent, delete=False, suffix=out_path.suffix) as tf:
            np.save(tf, combined)
            tmp = Path(tf.name)
        tmp.replace(out_path)
    else:
        np.save(out_path, combined)

    logging.info(f"Combined period matrix saved: {out_path} (files={len(files)})")
    return out_path

# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _write_event_summary_csv(events: List[Event], out_path: Path, safe_write: bool = True) -> Path:
    rows = [asdict(ev) for ev in events]
    df = pd.DataFrame(rows)
    ensure_directory_exists(out_path.parent)
    if safe_write:
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(out_path)
    else:
        df.to_csv(out_path, index=False)
    logging.info(f"Event summary CSV saved: {out_path}")
    return out_path

def _write_manifest_json(site: str, events: List[Event], paths: Dict[str, Any], out_path: Path) -> Path:
    payload = {
        "site": site,
        "events": [asdict(ev) for ev in events],
        "paths": {k: str(v) if isinstance(v, Path) else v for k, v in paths.items()}
    }
    ensure_directory_exists(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(out_path)
    logging.info(f"Manifest JSON saved: {out_path}")
    return out_path

# ---------------------------------------------------------------------------
# Diagnostics: missing timestamps (TXT) and dropped wet segments (CSV)
# ---------------------------------------------------------------------------

def _infer_step_seconds(ts: pd.Series) -> float:
    """Infer nominal step (seconds) from positive diffs; fallback: 60s."""
    dt = pd.to_datetime(ts, errors="coerce")
    diffs = dt.diff().dt.total_seconds().dropna()
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if not diffs.empty else 60.0

def summarize_missing_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a table of gaps with columns:
      start_time, end_time, gap_seconds, nominal_step_s, missing_count
    """
    if "Datetime" not in df.columns:
        return pd.DataFrame(columns=["start_time", "end_time", "gap_seconds", "nominal_step_s", "missing_count"])
    t = pd.to_datetime(df["Datetime"], errors="coerce").dropna().sort_values().reset_index(drop=True)
    if len(t) < 2:
        return pd.DataFrame(columns=["start_time", "end_time", "gap_seconds", "nominal_step_s", "missing_count"])
    step_s = _infer_step_seconds(t)
    diffs = t.diff().dt.total_seconds().fillna(0.0)
    gaps = diffs[diffs > 1.5 * step_s]  # bigger than 1.5× nominal step counts as a gap
    rows = []
    for i in gaps.index:
        start = t.iloc[i - 1]
        end = t.iloc[i]
        gap_s = float((end - start).total_seconds())
        miss = max(0, int(round(gap_s / step_s)) - 1)
        rows.append({
            "start_time": start,
            "end_time": end,
            "gap_seconds": gap_s,
            "nominal_step_s": step_s,
            "missing_count": miss,
        })
    return pd.DataFrame(rows)

def write_missing_timestamp_report_txt(
    df: pd.DataFrame,
    out_path: Path,
    *,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
) -> Path:
    """
    Write a TXT report with:
      - period dates
      - nominal step
      - expected vs observed samples in the period
      - total missing and percentage for the whole period
      - a per-gap listing
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "Datetime" not in df.columns:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("No 'Datetime' column available.\n")
        return out_path

    t_all = pd.to_datetime(df["Datetime"], errors="coerce").dropna().sort_values().reset_index(drop=True)
    if t_all.empty:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("No valid timestamps found.\n")
        return out_path

    step_s = _infer_step_seconds(t_all)

    if period_start and period_end:
        p0 = pd.to_datetime(period_start)
        p1 = pd.to_datetime(period_end)
    else:
        p0 = t_all.iloc[0]
        p1 = t_all.iloc[-1]

    mask = (t_all >= p0) & (t_all <= p1)
    t = t_all[mask].drop_duplicates()
    if t.empty:
        expected_count = int(((p1 - p0).total_seconds() // step_s) + 1)
        observed_count = 0
        missing_period = expected_count
    else:
        total_seconds = (p1 - p0).total_seconds()
        expected_count = int(round(total_seconds / step_s)) + 1
        observed_count = int(t.size)
        missing_period = max(expected_count - observed_count, 0)

    pct_missing = (missing_period / expected_count * 100.0) if expected_count > 0 else 0.0

    gaps_df = summarize_missing_timestamps(df)
    total_gap_missing = int(gaps_df["missing_count"].sum()) if not gaps_df.empty else 0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Missing Timestamp Report ===\n")
        f.write(f"Period: {p0}  →  {p1}\n")
        f.write(f"Nominal step (s): {int(step_s)}\n")
        f.write(f"Expected samples in period: {expected_count}\n")
        f.write(f"Observed unique timestamps: {observed_count}\n")
        f.write(f"Missing in period: {missing_period}\n")
        f.write(f"Missing percentage (period): {pct_missing:.3f}%\n")
        f.write(f"Sum of per-gap 'missing_count': {total_gap_missing}\n")
        f.write("\n--- Gaps (start, end, gap_seconds, missing_count) ---\n")
        if gaps_df.empty:
            f.write("No gaps detected.\n")
        else:
            for _, r in gaps_df.iterrows():
                f.write(f"{r['start_time']}, {r['end_time']}, {int(r['gap_seconds'])} s, miss={int(r['missing_count'])}\n")

    return out_path

def summarize_dropped_wet_segments(
    annotated: pd.DataFrame,
    *,
    min_duration_min: float,
    min_accum_mm: float,
    require_both_filters: bool,
) -> pd.DataFrame:
    """
    From the annotated dataframe (Datetime, event_id, is_raining, and either
    Depth_mm or accum_from_rate_mm), list wet segments that ended up with
    event_id == -1 and explain why they were dropped.
    """
    df = annotated.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Choose per-sample depth series
    if "Depth_mm" in df.columns:
        depth = pd.to_numeric(df["Depth_mm"], errors="coerce").fillna(0.0)
    elif "accum_from_rate_mm" in df.columns:
        depth = pd.to_numeric(df["accum_from_rate_mm"], errors="coerce").fillna(0.0)
    else:
        depth = integrate_rate_to_depth_mm(
            df, time_col="Datetime", rate_col="Intensidad", cap_gap_minutes=30
        ).fillna(0.0)

    wet = df["is_raining"].fillna(False).astype(bool).values
    evt = df["event_id"].fillna(-1).astype(int).values
    rate = pd.to_numeric(df.get("Intensidad", 0.0), errors="coerce").fillna(0.0)

    def _segments(mask: np.ndarray) -> List[Tuple[int, int]]:
        m = mask.astype(int)
        dm = np.diff(np.concatenate(([0], m, [0])))
        starts = np.where(dm == 1)[0]
        ends = np.where(dm == -1)[0] - 1
        return list(zip(starts, ends))

    rows = []
    for s, e in _segments(wet):
        if np.all(evt[s:e + 1] == -1):
            block_idx = slice(s, e + 1)
            start = df["Datetime"].iloc[s]
            end = df["Datetime"].iloc[e]
            dur_min = (end - start).total_seconds() / 60.0 if pd.notna(start) and pd.notna(end) else 0.0
            acc_mm = float(depth.iloc[block_idx].sum())
            pass_dur = dur_min >= float(min_duration_min)
            pass_acc = acc_mm >= float(min_accum_mm)

            if require_both_filters:
                reason_bits = []
                if not pass_dur:
                    reason_bits.append(f"duration<{min_duration_min}min")
                if not pass_acc:
                    reason_bits.append(f"accum<{min_accum_mm}mm")
                reason = " & ".join(reason_bits) if reason_bits else "kept"
            else:
                reason = "duration&&accum both failed" if (not pass_dur and not pass_acc) else "kept"

            rows.append({
                "start_time": start,
                "end_time": end,
                "duration_min": dur_min,
                "accum_mm": acc_mm,
                "n_samples": int(e - s + 1),
                "max_rate_mm_h": float(rate.iloc[block_idx].max()),
                "reason": reason,
            })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Orchestration helper used by scripts/event.py
# ---------------------------------------------------------------------------

def run_event_identification_for_site(
    df_processed: pd.DataFrame,
    *,
    cfg: Dict[str, Any],
    site: str,
    processed_site_dir: Path,
    events_site_dir: Path,
) -> Dict[str, Any]:
    """
    Identify events and write outputs according to config for a single site.
    Returns a summary dict with paths and counts.
    """
    ev_cfg = cfg.get("events", {}) or {}
    fnpat = (cfg.get("io", {}) or {}).get("filename_patterns", {}) or {}
    matrix_size_cfg = (cfg.get("processing", {}) or {}).get("parsivel", {}).get("matrix_size", [32, 32])
    matrix_size = int(matrix_size_cfg[0]) if isinstance(matrix_size_cfg, (list, tuple)) else int(matrix_size_cfg)

    # Thresholds & rules
    intensity_threshold = float(ev_cfg.get("intensity_threshold_mm_h", 0.1))
    start_thr = ev_cfg.get("start_threshold_mm_h")
    stop_thr = ev_cfg.get("stop_threshold_mm_h")
    min_wet_streak = int(ev_cfg.get("min_wet_streak", 1))
    min_dry_streak = int(ev_cfg.get("min_dry_streak", 1))
    max_gap_for_rate = ev_cfg.get("max_gap_for_rate_integration_min", None)
    max_gap_for_rate = float(max_gap_for_rate) if max_gap_for_rate not in (None, "") else None

    min_duration_min = int(ev_cfg.get("min_duration_min", 10))
    min_gap_min = int(ev_cfg.get("min_gap_min", 30))
    min_accum_mm = float(ev_cfg.get("min_accum_mm", 0.2))
    merge_if_gap_min = ev_cfg.get("merge_if_gap_min")
    merge_if_gap_min = int(merge_if_gap_min) if merge_if_gap_min not in (None, "") else None
    require_both_filters = bool(ev_cfg.get("require_both_filters", True))

    safe_write = bool((cfg.get("io", {}) or {}).get("safe_writes", True))

    # Identify events
    annotated, events = identify_precipitation_events(
        df_processed,
        intensity_threshold_mm_h=intensity_threshold,
        min_duration_min=min_duration_min,
        min_gap_min=min_gap_min,
        min_accum_mm=min_accum_mm,
        merge_if_gap_min=merge_if_gap_min,
        start_threshold_mm_h=start_thr,
        stop_threshold_mm_h=stop_thr,
        min_wet_streak=min_wet_streak,
        min_dry_streak=min_dry_streak,
        max_gap_for_rate_integration_min=max_gap_for_rate,
        require_both_filters=require_both_filters,
        site=site,
    )

    # Prepare dirs
    ensure_directory_exists(events_site_dir)
    matrices_dir = processed_site_dir / "matrices"

    # Save annotated and event CSVs
    annotated_path = save_annotated_data(annotated, events_site_dir, safe_write=safe_write)
    event_csv_pat = fnpat.get("event_csv", "event_{id}.csv")
    event_csv_paths = extract_and_save_events(annotated, events, events_site_dir, filename_pattern=event_csv_pat, safe_write=safe_write)

    # Save combined matrices according to config flags
    result: Dict[str, Any] = {
        "site": site,
        "annotated_csv": annotated_path,
        "event_csvs": event_csv_paths,
        "events": events,
        "combined_event_matrices": [],
        "combined_period_matrix": None,
        "summary_csv": None,
        "manifest_json": None,
    }

    out_flags = (ev_cfg.get("outputs", {}) or {})
    combine_mode = (out_flags.get("matrix_combine", "sum") or "sum").lower()

    if out_flags.get("save_per_event_combined", True):
        for ev in events:
            out_npy_name = fnpat.get("combined_npy", "combined_event_{id}.npy").format(id=ev.id)
            out_npy_path = events_site_dir / out_npy_name
            try:
                saved = save_combined_event_matrix(
                    matrices_dir=matrices_dir,
                    event=ev,
                    out_path=out_npy_path,
                    matrix_size=matrix_size,
                    safe_write=safe_write,
                    mode=combine_mode,
                )
                if saved:
                    result["combined_event_matrices"].append(saved)
            except Exception as e:
                logging.error(f"[{site}] Failed to save combined matrix for event {ev.id}: {e}", exc_info=True)

    if out_flags.get("save_period_combined", True):
        ts = pd.to_datetime(df_processed["Datetime"], errors="coerce")
        ts_min = int(ts.min().timestamp())
        ts_max = int(ts.max().timestamp())
        out_name = fnpat.get("combined_period_npy", "combined_period_{start}_{end}.npy").format(
            start=pd.to_datetime(ts_min, unit="s").strftime("%Y%m%d%H%M%S"),
            end=pd.to_datetime(ts_max, unit="s").strftime("%Y%m%d%H%M%S"),
        )
        out_path = events_site_dir / out_name
        try:
            result["combined_period_matrix"] = save_combined_period_matrix(
                matrices_dir=matrices_dir,
                start_ts=ts_min,
                end_ts=ts_max,
                out_path=out_path,
                matrix_size=matrix_size,
                safe_write=safe_write,
                mode=combine_mode,
            )
        except Exception as e:
            logging.error(f"[{site}] Failed to save combined period matrix: {e}", exc_info=True)

    # Summaries
    try:
        result["summary_csv"] = _write_event_summary_csv(events, events_site_dir / "event_summary.csv", safe_write=safe_write)
        result["manifest_json"] = _write_manifest_json(
            site=site,
            events=events,
            paths={
                "annotated": annotated_path,
                "event_csvs": [str(p) for p in event_csv_paths],
                "combined_event_npys": [str(p) for p in result["combined_event_matrices"]],
                "combined_period_npy": str(result["combined_period_matrix"]) if result["combined_period_matrix"] else None,
            },
            out_path=events_site_dir / "manifest.json",
        )
    except Exception as e:
        logging.error(f"[{site}] Failed to write summaries: {e}", exc_info=True)

    # Missing timestamps (TXT) — period-level percentage using config dates if present
    try:
        period_start = str((cfg.get("start_date") or "")) or None
        period_end = str((cfg.get("end_date") or "")) or None
        miss_txt = events_site_dir / "missing_timestamps.txt"
        write_missing_timestamp_report_txt(
            df_processed,
            miss_txt,
            period_start=period_start,
            period_end=period_end,
        )
        logging.info(f"[{site}] Missing timestamps report saved: {miss_txt}")
    except Exception as e:
        logging.error(f"[{site}] Failed to write missing_timestamps.txt: {e}", exc_info=True)

    # Dropped wet segments diagnostics
    try:
        drop_df = summarize_dropped_wet_segments(
            annotated,
            min_duration_min=float(ev_cfg.get("min_duration_min", 10)),
            min_accum_mm=float(ev_cfg.get("min_accum_mm", 0.2)),
            require_both_filters=bool(ev_cfg.get("require_both_filters", True)),
        )
        drop_csv = events_site_dir / "dropped_wet_segments.csv"
        drop_df.to_csv(drop_csv, index=False)
        logging.info(f"[{site}] Dropped wet segments: {len(drop_df)} (saved {drop_csv})")

        mask = (annotated["is_raining"] == True) & (annotated["event_id"] == -1)
        raw_csv = events_site_dir / "wet_but_not_in_event_samples.csv"
        annotated.loc[mask].to_csv(raw_csv, index=False)
        logging.info(f"[{site}] Wet samples not in event: {mask.sum()} (saved {raw_csv})")
    except Exception as e:
        logging.error(f"[{site}] Diagnostics export failed: {e}", exc_info=True)

    logging.info(f"[{site}] Events found: {len(events)}")
    return result