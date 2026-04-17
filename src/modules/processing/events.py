import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from ..config import load_config
from ..io_utils import ensure_dir, safe_write_csv

def identify_events(input_csv: Path, out_dir: Path, cfg: Dict[str, Any], site: str) -> Dict[str, Any]:
    """
    Stage 2: Canonical CSV -> Rain Events.
    Uses unified logic:
    - Load CSV (with ISO datetime).
    - Convert to local timezone if needed? Or stick to UTC/Input ref?
      Decision: Config says 'timezone', we should enforce it here.
    - Segment periods where Intensidad > threshold (0.1 mm/h).
    - Apply Inter-Event Time (30 min).
    - Filter minimal events (depth > 0.1 mm).
    """
    
    # 1. Load Data
    if not input_csv.exists():
        logging.error(f"Input CSV not found: {input_csv}")
        return {"events_count": 0, "error": "InputNotFound"}
        
    df = pd.read_csv(input_csv)
    if df.empty:
        return {"events_count": 0}
        
    # 2. Prepare Columns
    # Expect: Timestamp, Datetime, Intensidad, Precipitacion
    # Ensure sorted
    df = df.sort_values("Timestamp").reset_index(drop=True)
    
    # Timezone Handling
    target_tz = cfg["parsing"]["timezone"]
    
    # Convert 'Datetime' str to actual datetime objects
    # If they are naive, assume they match the target_tz? 
    # Or assume input was UTC? 
    # VST is usually Local Time. DSD2 is Epoch (UTC?).
    # Ideally Stage 1 normalized this. 
    # For now, let's treat 'Datetime' column as the reference truth.
    df['dt_obj'] = pd.to_datetime(df['Datetime'])
    
    # 3. Apply Thresholds
    intensity_th = cfg["events"]["intensity_threshold"] # 0.1
    inter_event_min = cfg["events"]["inter_event_minutes"] # 30
    min_depth = cfg["events"]["min_event_depth_mm"] # 0.1
    
    # Boolean mask: Is raining?
    # Use fillna(0) to treat NaNs as dry
    is_raining = df["Intensidad"].fillna(0) > intensity_th
    
    # 4. Segmentation Logic
    # Group consecutive raining periods, bridging gaps < inter_event_min
    
    # We need a numeric time column in minutes to easy calc
    # Timestamp is seconds.
    time_min = df["Timestamp"] / 60.0
    
    events = []
    
    # Find indices where it is raining
    rain_indices = df.index[is_raining].tolist()
    
    if not rain_indices:
        logging.info("No rain detected above threshold.")
        return {"events_count": 0}
        
    # Iterate and cluster
    current_event = [rain_indices[0]]
    
    for i in range(1, len(rain_indices)):
        prev_idx = rain_indices[i-1]
        curr_idx = rain_indices[i]
        
        # Check gap in MINUTES (using Timestamp column)
        gap_min = time_min[curr_idx] - time_min[prev_idx]
        
        if gap_min <= inter_event_min:
             # Valid gap, continue event
             # Note: This includes the dry period in between as part of the event?
             # Standard definition: "Event includes the dry gap if < 30min".
             # So we should include all indices between prev and curr?
             # Yes.
             # range(prev_idx + 1, curr_idx) are dry rows inside the event.
             current_event.extend(range(prev_idx + 1, curr_idx + 1))
        else:
             # End of event
             events.append(sorted(list(set(current_event))))
             current_event = [curr_idx]
             
    # Append last
    if current_event:
        events.append(sorted(list(set(current_event))))
        
    # 5. Filter & Export
    ensure_dir(out_dir)
    
    valid_events_summary = []
    event_counter = 1
    
    for evt_indices in events:
        evt_df = df.iloc[evt_indices]
        
        # Calc Metrics
        # Duration: last - first (minutes)
        duration_min = (evt_df["Timestamp"].iloc[-1] - evt_df["Timestamp"].iloc[0]) / 60.0
        if duration_min < 0: duration_min = 0 # Should not happen
        
        # Total Precip
        # User Requirement: "the column precipitacion is not usefull it is better not to use it"
        # We ALWAYS measure depth by integrating Intensity (mm/h) * time.
        
        evt_intensities = evt_df["Intensidad"].values
        # Simple integration: sum(Intensity_mm_h) / 60 
        # Assuming 1-minute steps (standard DSD).
        
        precip_col_sum = np.sum(evt_intensities) / 60.0
        
        # Basic filter
        if precip_col_sum < min_depth:
            continue
            
        # Valid Event -> Save
        start_dt = evt_df["Datetime"].iloc[0]
        end_dt = evt_df["Datetime"].iloc[-1]
        
        filename = f"event_{event_counter}.csv"
        evt_path = out_dir / filename
        evt_df.to_csv(evt_path, index=False)
        
        valid_events_summary.append({
            "Event_ID": event_counter,
            "Start_Date": start_dt,
            "End_Date": end_dt,
            "Duration_Min": round(duration_min, 2),
            "Total_Precip_mm": round(precip_col_sum, 2),
            "Max_Intensity": evt_df["Intensidad"].max(),
            "Filename": filename
        })
        
        event_counter += 1
        
    # Save Summary
    summary_path = out_dir / "event_summary.csv"
    if valid_events_summary:
        safe_write_csv(summary_path, valid_events_summary, valid_events_summary[0].keys())
        logging.info(f"Identified {len(valid_events_summary)} valid events for {site}.")
    else:
        logging.info(f"No valid events (depth > {min_depth}mm) for {site}.")
        
    return {
        "site": site,
        "events_count": len(valid_events_summary),
        "summary_path": str(summary_path) if valid_events_summary else None
    }
