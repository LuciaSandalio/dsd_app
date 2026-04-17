import os
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# Define paths
RAW_DIR = Path("data/raw")

# Date pattern parser (reusing logic or simple regex)
def parse_date(filename):
    # Try YYYYMMDD
    match = re.search(r"(\d{8})", filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d")
        except:
            pass
    # Try YYYY-MM-DD
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d")
    return None

def detect_format(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(3): # Check first few lines
                line = f.readline()
                if not line: break
                if ";" in line:
                    return "VST_Semicolon"
                if ":" in line:
                    parts = line.split(":")
                    if len(parts) > 5:
                        return "Standard_Colon"
    except:
        return "Error"
    return "Unknown"

data = []

# Extract site (parent directory of the file's parent usually, or direct parent depending on structure)
# Structure: data/raw/site/month/day/file.txt -> parts[-4] is site
def get_site(path_obj):
    try:
        # relative to data/raw?
        # path is data/raw/site/month/day/file
        # parts: data, raw, site, ...
        # reliable way: relative to data/raw, then take first part
        rel = path_obj.relative_to(RAW_DIR)
        return rel.parts[0]
    except:
        return "unknown"

print(f"Scanning {RAW_DIR}...")
for root, dirs, files in os.walk(RAW_DIR):
    for name in files:
        if not name.endswith(".txt"): continue
        
        dt = parse_date(name)
        if not dt: continue
        
        path = Path(root) / name
        fmt = detect_format(path)
        site = get_site(path)
        data.append({"date": dt, "file": name, "format": fmt, "path": str(path), "site": site})

if not data:
    print("No files found.")
    exit()

df = pd.DataFrame(data)
df = df.sort_values("date")

print("\n--- Format Timeline by Site ---")
sites = df['site'].unique()

for site in sorted(sites):
    print(f"\n[Site: {site}]")
    site_df = df[df['site'] == site]
    
    # Group by format
    formats_seen = site_df['format'].unique()
    for fmt in formats_seen:
        subset = site_df[site_df['format'] == fmt]
        if subset.empty: continue
        min_date = subset['date'].min().strftime("%Y-%m-%d")
        max_date = subset['date'].max().strftime("%Y-%m-%d")
        print(f"  Format: {fmt: <15} | Range: {min_date} -> {max_date} | Files: {len(subset)}")
    
    # Check for transition
    vst = site_df[site_df['format'] == "VST_Semicolon"]
    std = site_df[site_df['format'] == "Standard_Colon"]
    
    if not vst.empty and not std.empty:
        last_std = std['date'].max()
        first_vst = vst['date'].min()
        print(f"  -> Transition detected!")
        print(f"     Last Standard: {last_std}")
        print(f"     First VST:     {first_vst}")
    elif not vst.empty:
        print("  -> Always VST (in this dataset)")
    elif not std.empty:
        print("  -> Always Standard (in this dataset)")

# Detailed transition zone
print("\n--- Transition Details ---")
# Look for the boundary where format switches
# Naive approach: check if dates overlap or distinct
dates_vst = df[df['format'] == "VST_Semicolon"]['date']
dates_std = df[df['format'] == "Standard_Colon"]['date']

if not dates_vst.empty and not dates_std.empty:
    last_std = dates_std.max()
    first_vst = dates_vst.min()
    print(f"Last Standard File: {last_std}")
    print(f"First VST File:     {first_vst}")
    
    if first_vst < last_std:
        print("WARNING: Formats overlap in time.")
    else:
        print("Clear transition detected.")
else:
    print("Only one format detected or insufficient data.")
