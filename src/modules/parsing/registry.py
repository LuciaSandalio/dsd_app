from typing import Tuple, List, Optional
from ..models import SampleRecord, ParseStats, DetectedFormat, FormatType
from .detect import detect_format_from_file
from .dsd2_packed import Dsd2PackedColonParser
from .vst_semicolon import VstSemicolonParser

def parse_file_auto(filepath: str) -> Tuple[List[SampleRecord], ParseStats]:
    """Auto-detect format and parse file."""
    
    detection = detect_format_from_file(filepath)
    fmt = detection.fmt
    
    parser = None
    if fmt == FormatType.DSD2_PACKED:
        parser = Dsd2PackedColonParser()
    elif fmt == FormatType.VST_SEMICOLON:
        parser = VstSemicolonParser()
    elif fmt == FormatType.DSD1_SEMICOLON:
        # Use VST parser for now as they are similar (semicolon family)
        # or implement specific one if needed.
        parser = VstSemicolonParser() 
    else:
        return [], ParseStats(filepath, "unknown", errors={"DetectionFailed": detection.details})
        
    return parser.parse(filepath)
