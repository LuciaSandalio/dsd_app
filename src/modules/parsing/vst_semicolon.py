import logging
import numpy as np
from typing import List, Tuple
from datetime import datetime
from ..models import SampleRecord, ParseStats

class VstSemicolonParser:
    """
    Parses 'New' format variations (VST / DSD1 Semicolon).
    Common trait: Semicolons separation, usually ~1024 separate fields for matrix.
    Logic: Strict index-based parsing to avoid data pollution from header fields.
    """
    
    def parse(self, filepath: str) -> Tuple[List[SampleRecord], ParseStats]:
        stats = ParseStats(filepath, "vst_semicolon")
        records = []
        
        try:
            with open(filepath, 'r', encoding='latin-1', errors='replace') as f:
                lines = f.readlines()
                stats.total_lines = len(lines)
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        stats.skipped_lines += 1
                        continue
                        
                    try:
                        record = self._parse_line(line)
                        if record:
                            records.append(record)
                            stats.parsed_lines += 1
                            if record.matrix is not None:
                                stats.matrix_count += 1
                        else:
                            stats.skipped_lines += 1
                    except:
                         stats.skipped_lines += 1
                         
        except Exception as e:
            stats.errors["ReadError"] = str(e)
            
        return records, stats

    def _parse_line(self, line: str) -> SampleRecord:
        # Strict parsing based on DSD1/VST Semicolon structure
        # Expected: Timestamp Val0;Val1;...;ValX;[Matrix 1024];[Nd 32];[Vd 32];...
        
        parts = line.split(";")
        if len(parts) < 1030: # Need at least header + 1024 matrix
            return None
            
        # 1. Parse Header & Timestamp
        # part[0] is typically "Epoch Intensity" or "Date Time Intensity"
        p0 = parts[0].strip()
        p0_tokens = p0.split()
        
        timestamp = 0
        datetime_str = ""
        intensity = 0.0
        reflectivity = 0.0
        status = 0
        particles_count = 0
        
        try:
            # Check first token
            token0 = p0_tokens[0]
            if token0.isdigit() and len(token0) >= 9: # Epoch
                timestamp = int(token0)
                datetime_str = datetime.fromtimestamp(timestamp).isoformat()
                # Intensity is the next token(s) join, but usually just one float
                if len(p0_tokens) > 1:
                    intensity = float(p0_tokens[1])
            elif ":" in line[:20]: # ISO-like Date
                # Heuristic: First 19 chars are YYYY-MM-DD HH:MM:SS
                # Re-parse p0 string.
                # "2025-08-19 01:42:00 0.569"
                if len(p0_tokens) >= 2:
                    ts_str = " ".join(p0_tokens[:2])
                    try:
                        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                        timestamp = int(dt.timestamp())
                        datetime_str = dt.isoformat()
                        if len(p0_tokens) > 2:
                            intensity = float(p0_tokens[2])
                    except:
                        pass
        
            # Parse other header fields from parts[1]...
            # Map based on legacy: 
            # 0: Int (handled), 1: Interval, 2: Count, 3: Code, 4: KinE, 5: Refl, 6: Status
            if len(parts) > 6:
                if not intensity and len(parts) > 0 and parts[0].replace('.','',1).isdigit():
                     # Fallback if p0 parsing failed to grab intensity
                     pass
                
                try:
                    reflectivity = float(parts[5])
                    status = int(parts[6])
                    particles_count = int(parts[2])
                except: pass
                
        except Exception:
            return None

        # 2. Extract Matrix (Strict Index 7 to 1031)
        try:
            # We strictly take the dedicated matrix fields
            # Legacy parser: raw_matrices.append(list(map(float, parts[7:7+1024])))
            matrix_cells = parts[7:1031]
            
            # Convert to ints. Use float->int safely.
            matrix_data = []
            for c in matrix_cells:
                c = c.strip()
                if not c: 
                    matrix_data.append(0)
                    continue
                try:
                    # Parse as float then cast to int to handle "0.0" or "0"
                    matrix_data.append(int(float(c)))
                except:
                    matrix_data.append(0)
            
            # Ensure exactly 1024 elements
            if len(matrix_data) != 1024:
                if len(matrix_data) < 1024:
                    matrix_data += [0] * (1024 - len(matrix_data))
                else:
                    matrix_data = matrix_data[:1024]

            # Enforce 32x32 reshape
            matrix = np.array(matrix_data, dtype=np.int32).reshape(32, 32)
            
            return SampleRecord(
                timestamp=timestamp,
                datetime_str=datetime_str,
                intensity=intensity,
                manager_intensity=intensity,
                reflectivity=reflectivity,
                status=status,
                particles_count=particles_count,
                matrix=matrix,
                precip_accum=None
            )
            
        except Exception:
            return None
