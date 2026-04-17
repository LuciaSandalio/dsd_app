import re
from typing import List, Tuple
from ..models import DetectedFormat, FormatType

# Regex patterns
# DSD2 Packed: Start with 10 digits (epoch), followed by space, then digits:digits...
# Example: 1735866000 00000:023:026...
RE_DSD2_PACKED = re.compile(r"^\d{9,10}\s+\d+:")

# DSD1 Semicolon: Start with 10 digits (epoch), followed by space, then digits;digits...
# Example: 1765731600 0000.000;00060...
RE_DSD1_SEMICOLON = re.compile(r"^\d{9,10}\s+[\d\.]+[;]")

# VST Semicolon: Start with YYYY-MM-DD, maybe time, then separator
# Example: 2025-01-25 11:04:01: ... or 2025-01-25 11:04:01 ...
RE_VST_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")

def detect_format_from_file(filepath: str, num_lines: int = 3) -> DetectedFormat:
    """Read first N lines and detect format."""
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(num_lines * 3): # Read extra to skip blanks
                line = f.readline()
                if not line: break
                line = line.strip()
                if line:
                    lines.append(line)
                    if len(lines) >= num_lines: break
    except Exception as e:
        return DetectedFormat(FormatType.UNKNOWN, 0.0, f"Read error: {e}")

    if not lines:
        return DetectedFormat(FormatType.UNKNOWN, 0.0, "Empty file")

    # Vote on format
    votes = {fmt: 0 for fmt in FormatType}
    
    for line in lines:
        fmt = _detect_line(line)
        votes[fmt] += 1
        
    # Get winner
    best_fmt = max(votes, key=votes.get)
    confidence = votes[best_fmt] / len(lines)
    
    return DetectedFormat(best_fmt, confidence, f"Votes: {votes}")

def _detect_line(line: str) -> FormatType:
    # Check for VST Date first (very distinct)
    if RE_VST_DATE.match(line):
        return FormatType.VST_SEMICOLON

    # Check for Epoch start
    if RE_DSD2_PACKED.match(line):
         # Double check explicit colon count vs semicolon?
         if ":" in line and ";" not in line:
             return FormatType.DSD2_PACKED
    
    if RE_DSD1_SEMICOLON.match(line):
        return FormatType.DSD1_SEMICOLON
        
    # Fallback heuristics if regex misses (e.g. malformed start)
    semi_count = line.count(";")
    colon_count = line.count(":")
    
    if semi_count > 10:
        # Heavily semicolon -> Likely VST or DSD1
        # If it has a date-like start -> VST, else DSD1
        if RE_VST_DATE.search(line): 
            return FormatType.VST_SEMICOLON
        return FormatType.DSD1_SEMICOLON
        
    if colon_count > 5:
        return FormatType.DSD2_PACKED
        
    return FormatType.UNKNOWN
