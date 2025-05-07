# scripts/master_workflow.py

import sys
from pathlib import Path
import subprocess
import argparse
import yaml
import logging
import pandas as pd

# Correctly determine project root by going up three levels
project_root = Path(__file__).parent.parent.parent.resolve()
print(f"Project root: {project_root}")

# Insert project_root into sys.path if not already present
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Inserted {project_root} into sys.path")

# Import from modules after setting up sys.path and logging
from modules.utils import cleanup_output, configure_logging, load_config

from scripts.get_dsdfile import get_dsdfile_main
from scripts.event import event_main
from scripts.visualization_dsd import visualization_main

def run_script(script_path, args=[]):
    """
    Runs a Python script with the given arguments.
    """
    try:
        command = [sys.executable, script_path] + args
        logging.info(f"Running script: {' '.join(command)}")
        subprocess.run(command, check=True)
        logging.info(f"Successfully ran {script_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_path}: {e}")
        sys.exit(1)

def update_config_with_dates(config_path, start_date, end_date):
    """
    Updates the config.yaml file with the provided start and end dates.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        config['get_dsdfile']['start_date'] = start_date
        config['get_dsdfile']['end_date'] = end_date

        config['event']['start_date'] = start_date
        config['event']['end_date'] = end_date

        config['visualization']['start_date'] = start_date
        config['visualization']['end_date'] = end_date

        with open(config_path, 'w') as file:
            yaml.dump(config, file)

        logging.info(f"Updated config.yaml with start_date: {start_date} and end_date: {end_date}")
    except Exception as e:
        logging.critical(f"Failed to update config.yaml: {e}")
        sys.exit(1)

def validate_dates(start_date, end_date):
    """
    Validates the format and logical order of the provided dates.
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if start > end:
            logging.error("Start date must be earlier than or equal to end date.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Date validation error: {e}")
        sys.exit(1)

def run_master_workflow(start_date: str, end_date: str, config_path: str(project_root / "config" / "config.yaml")) -> None: # type: ignore
    """
    Executes the master workflow for a given date range.
    
    Parameters:
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - config_path (str): Path to the configuration YAML file.
    """
    # Configure logging using the centralized function
    try:
        config = load_config(Path(config_path).resolve())
        log_file_path = config.get('workflow', {}).get('log_file', 'logs/master_workflow.log')
    except Exception as e:
        print(f"Failed to load configuration for logging: {e}")
        sys.exit(1)

    configure_logging(log_file_path)

    logging.info("Master Workflow Started.")

    # Update config.yaml with start and end dates
    validate_dates(start_date, end_date)
    update_config_with_dates(config_path, start_date, end_date)

    # Reload configuration after updating
    try:
        config = load_config(Path(config_path).resolve())
    except Exception as e:
        logging.error(f"Failed to reload configuration after updating dates: {e}")
        sys.exit(1)

    # Step 0: Cleanup previous outputs
    try:
        cleanup_output(config)
        logging.info("Cleanup of previous outputs completed.")
    except Exception as e:
        logging.critical(f"Failed during cleanup: {e}")
        sys.exit(1)

    get_dsdfile_main(start_date, end_date, config_path)
    event_main(start_date, end_date, config_path)
    visualization_main(start_date, end_date, config_path)

    logging.info("Master Workflow Completed Successfully.")
    print("Master workflow completed successfully.")


