import logging
import numpy as np
from typing import List, Tuple
from datetime import datetime
from ..models import SampleRecord, ParseStats
from ..io_utils import ensure_dir

class Dsd2PackedColonParser:
    """
    Parses 'Standard/Old' format:
    Timestamp Value1:Value2:...:PackedBlob
    delimiter = :
    PackedBlob = long string of integers without delimiters or with odd fixed width.
    Actually, per previous analysis: "1735866000 00000:023:026..."
    And the blob is space, dot or colon separated? 
    Wait, the user's prompt said "dsd2_packed (Colon + Packed Blob)".
    
    Strategy:
    1. Split by space -> Timestamp, Rest
    2. Split Rest by ':'
    3. Last part is the Matrix Blob? Or the Matrix is split by colons too?
    
    Re-reading previous analysis:
    DataProcessing.py said: 
    if len(parts_rest) > 100: Case Matrix cells are split by delimiters (VST).
    Else (len < 100): Standard Colon.
    
    In Standard Colon, matrix is usually NOT fully split by colons. It's often a single blob or chunks.
    Let's implement a robust "last 1024 ints" extraction like we did for VST, 
    but tuned for Colon files.
    """
    
    def parse(self, filepath: str) -> Tuple[List[SampleRecord], ParseStats]:
        stats = ParseStats(filepath, "dsd2_packed")
        records = []
        
        try:
            with open(filepath, 'r', encoding='latin-1', errors='replace') as f:
                lines = f.readlines()
                stats.total_lines = len(lines)
                
                for i, line in enumerate(lines):
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
                    except Exception as e:
                        # logging.debug(f"Line {i} error: {e}")
                        stats.errors[str(e)] = stats.errors.get(str(e), 0) + 1
                        stats.skipped_lines += 1
                        
        except Exception as e:
            stats.errors["FileReadError"] = str(e)
            
        return records, stats

    def _parse_line(self, line: str) -> SampleRecord:
        # Structure: Epoch rest
        tokens = line.split(maxsplit=1)
        if len(tokens) < 2: return None
        
        epoch_str = tokens[0]
        rest = tokens[1]
        
        if not epoch_str.isdigit(): return None
        timestamp = int(epoch_str)
        
        # Split fields by colon
        parts = rest.split(":")
        
        # Extract fields (assuming typical dsd2 map)
        # 0: Intensity (or similar)
        # But wait, mappings differ.
        # Let's assume standard mapping:
        # [0] Intensity? No, standard internal schema was:
        # [0] NumPart, [1] Temp, [2] HeadR, [3] HeadL, [4] Heater, [5] Intensity, [6] Precip, [7] Refl, [8] Status
        # BUT this depends entirely on the file.
        
        # Heuristic: Parse all floats
        # If Matrix is in there, we need to separate it.
        # DSD2 Packed usually means the matrix is ONE field or few fields.
        
        # Let's define specific logic for "Pilar Old" style:
        # 00000:023:026:026:0.16:0000.000:0000.00:-9.999:0:0000000...
        
        # parts[0]: 00000 (Sample Int?)
        # parts[1]: 023 (Particles?)
        # parts[2]: 026 (Temp?)
        # ...
        
        # Crucial fields:
        # Intensity: Look for mm/h (often 0000.000)
        # Reflectivity: dBZ (often -9.999)
        
        # For this refactor, I will use a conservative mapping based on index.
        # Assuming header size ~9 fields.
        
        if len(parts) < 9: return None
        
        # Mapping (Best Guess from legacy code)
        try:
            # Legacy Schema: [NumPart, Temp, HeadR, HeadL, Heater, Intensity, Precip, Refl, Status]
            # Map parts to this?
            # Let's parse strictly 9 head fields, then rest is matrix.
            
            # parts[0]: ?
            # ...
            # Actually, `data_processing.py` didn't explicitly remap the "Standard" format,
            # it just took `parts[:9]`. 
            # So we assume the file IS in that Internal Schema order?
            # OR the legacy code mapped it 1:1.
            
            # Let's go with 1:1 for now.
            
            # Fields
            num_particles = int(parts[0])
            temp_housing = float(parts[1])
            # ... skipping extended sensors for brevity/safety unless crucial
            intensity = float(parts[5])
            precip = float(parts[6])
            reflectivity = float(parts[7])
            status = int(parts[8])

            # Matrix: Part 9 is the blob?
            raw_blob = "".join(parts[9:]) # Join rest
            # In "Packed", digits are fixed width (3 chars). 32x32=1024 cells * 3 chars = 3072 chars.
            
            matrix = None
            if len(raw_blob) >= 3072:
                # Chunk into 3s
                cells = [int(raw_blob[i:i+3]) for i in range(0, 3072, 3)]
                matrix = np.array(cells, dtype=np.int32).reshape(32, 32)
                
            return SampleRecord(
                timestamp=timestamp,
                datetime_str=datetime.fromtimestamp(timestamp).isoformat(),
                intensity=intensity,
                manager_intensity=intensity,
                precip_accum=precip,
                reflectivity=reflectivity,
                status=status,
                particles_count=num_particles,
                temp_housing=temp_housing,
                matrix=matrix
            )
            
        except ValueError:
            return None
