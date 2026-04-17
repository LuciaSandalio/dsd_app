#!/usr/bin/env python3
# src/scripts/master_run.py
# Unified entrypoint for DSD Processing Pipeline.
# Runs Stage 1 -> Stage 2 -> Stage 3 sequentially.

import sys
import argparse
import logging
from pathlib import Path
import subprocess

# Ensure project root in path
_THIS = Path(__file__).resolve()
PROJ_ROOT = _THIS.parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# Import CLI main functions directly to avoid subprocess overhead?
# We can import `main` from each script if we structure them to accept args or config object.
# But current scripts parse sys.argv.
# It is safer/easier to run them as subprocess calls or simulate args?
# Let's import the `main` functions but we need to patch sys.argv or refactor them to take args.
# Refactoring them to take args is cleaner but might be too much change right now.
# Subprocess is robust and isolates output.
# HOWEVER, reusing python process is better for logging.

# Let's try importing, but we need to ensure they don't rely solely on argparse from sys.argv.
# I will use subprocess for maximum isolation and simplicity as a "Master Runner" script.
# This mimics "putting them in a shell script" but with Python control.

import datetime

# Configure Logging
# We want: 
# 1. Console Output (for GUI/User)
# 2. File Output (for persistence in logs/)

# Create logs directory
LOG_DIR = PROJ_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"run_{timestamp}.log"

# Get root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s [MASTER] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# File Handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream Handler (Console)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def run_step(script_name: str, args: list):
    """Run a script as a subprocess and capture output to log."""
    script_path = PROJ_ROOT / "src" / "scripts" / script_name
    cmd = [sys.executable, str(script_path)] + args
    
    logging.info(f"--- Starting {script_name} ---")
    logging.info(f"Command: {' '.join(cmd)}")
    
    # Run process, capturing output
    try:
        # We use Popen to stream output line by line
        with subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, # Merge stderr to stdout
            text=True, 
            bufsize=1, 
            universal_newlines=True
        ) as p:
            # Read stdout line by line
            for line in p.stdout:
                line = line.strip()
                if line:
                    # Log to file and console via our logger
                    # Note: Subprocess logs likely have their own timestamp/format.
                    # To avoid double timestamps, we can just log the message.
                    # But our formatter adds [MASTER] timestamp.
                    # Option 1: Just print to stdout (skip duplicate log to console) and log to file?
                    # Option 2: Log everything as INFO.
                    logging.info(f"[{script_name}] {line}")
            
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd)
            
        logging.info(f"--- Finished {script_name} (Exit: {p.returncode}) ---\n")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"!!! Failed {script_name} (Exit: {e.returncode}) !!!")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Master DSD Workflow")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--site", help="Run for specific site only")
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all", help="Run specific stage or all")
    parser.add_argument("--start-date", help="YYYY-MM-DD (Stage 1 only)")
    parser.add_argument("--end-date", help="YYYY-MM-DD (Stage 1 only)")
    
    args = parser.parse_args()
    
    # Propagate common args
    common_args = ["--config", args.config]
    if args.site:
        common_args += ["--site", args.site]
        
    # Stage 1: Get Data (Merge)
    if args.stage in ["1", "all"]:
        s1_args = common_args.copy()
        if args.start_date: s1_args += ["--start-date", args.start_date]
        if args.end_date: s1_args += ["--end-date", args.end_date]
        run_step("get_dsdfile.py", s1_args)
        
    # Stage 2: Event ID
    if args.stage in ["2", "all"]:
        run_step("event.py", common_args)
        
    # Stage 3: Parameterization
    if args.stage in ["3", "all"]:
        run_step("parameterize.py", common_args)
        
    logging.info("=== Workflow Completed Successfully ===")
    print(f"\nSee detailed log at: {log_file}")

if __name__ == "__main__":
    main()
