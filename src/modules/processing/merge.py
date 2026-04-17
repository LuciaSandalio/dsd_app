import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor

from ..models import IngestResult, SampleRecord, ParseStats
from ..parsing.registry import parse_file_auto
from ..io_utils import safe_write_csv, ensure_dir, atomic_write

def merge_site_data(raw_files: List[Path], out_csv: Path, out_matrices_dir: Path, cfg: Dict[str, Any], site: str) -> IngestResult:
    """
    Stage 1: Ingest raw files -> Canonical CSV + NPY Matrices.
    """
    total_rows = 0
    matrices_saved = 0
    files_processed = 0
    
    all_records: List[Dict[str, Any]] = []
    
    # Manifest data
    manifest = {
        "site": site,
        "config_snapshot": cfg.get("parsing", {}),
        "files": []
    }
    
    # Ensure dirs
    ensure_dir(out_matrices_dir)
    ensure_dir(out_csv.parent)
    ensure_dir(out_csv.parent / "diagnostics")
    
    # Process files
    # Only process sorted list to ensure deterministic order if logic depends on it
    # (Though we sort by timestamp later)
    for raw_file in sorted(raw_files):
        logging.info(f"Processing {raw_file}")
        
        try:
            records, stats = parse_file_auto(str(raw_file))
            files_processed += 1
            
            # Save stats to manifest
            manifest["files"].append({
                "path": str(raw_file),
                "format": stats.format,
                "lines": stats.total_lines,
                "parsed": stats.parsed_lines,
                "skipped": stats.skipped_lines,
                "errors": stats.errors
            })
            
            for rec in records:
                # Save Matrix if present
                if rec.matrix is not None:
                    mat_path = out_matrices_dir / f"matrix_{rec.timestamp}.npy"
                    np.save(mat_path, rec.matrix)
                    matrices_saved += 1
                
                # Convert to dict for CSV
                row = {
                    "Timestamp": rec.timestamp,
                    "Datetime": rec.datetime_str,
                    "Intensidad": rec.intensity,
                    "Precipitacion": rec.precip_accum if rec.precip_accum is not None else np.nan,
                    "Reflectividad": rec.reflectivity if rec.reflectivity is not None else np.nan,
                    "Estado": rec.status,
                    "Cant_Particulas": rec.particles_count if rec.particles_count is not None else 0
                    # Add other fields as needed
                }
                all_records.append(row)
                
        except Exception as e:
            logging.error(f"Failed to process {raw_file}: {e}")
            manifest["files"].append({"path": str(raw_file), "error": str(e)})

    # Create Canonical DataFrame
    df = pd.DataFrame(all_records)
    
    if not df.empty:
        # Deduplicate: Keep LAST by Timestamp
        df = df.sort_values("Timestamp")
        df = df.drop_duplicates(subset=["Timestamp"], keep="last")
        
        total_rows = len(df)
        
        # Save CSV
        # Use pandas to_csv but atomically? 
        # Or use our safe_write_csv if we convert back to dict?
        # Pandas is easier for DF.
        temp_csv = str(out_csv) + ".tmp"
        df.to_csv(temp_csv, index=False)
        import shutil
        shutil.move(temp_csv, out_csv)
    else:
        logging.warning("No valid records found in any file.")

    # Save Manifest
    manifest_path = out_csv.parent / "diagnostics" / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    return IngestResult(
        site=site,
        files_processed=files_processed,
        total_rows=total_rows,
        matrices_saved=matrices_saved,
        output_csv=str(out_csv),
        manifest_path=str(manifest_path)
    )
