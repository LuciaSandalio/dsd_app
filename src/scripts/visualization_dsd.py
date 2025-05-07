#!/usr/bin/env python3
# visualization_dsd.py
# Generates visualization plots for precipitation events

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# Ensure project root is on sys.path for module imports
project_root = Path(__file__).parent.parent.parent.resolve()  
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Import custom modules
from modules.utils import (
    setup_initial_logging,
    configure_logging,
    load_config,
    ensure_directory_exists
)
from modules.visualization import (
    resource_path,
    create_plots_output_directories,
    load_diam_vel_mapping_csv,
    plot_precipitation_intensity_separate,
    plot_hyetograph,
    plot_accumulated_precipitation,
    plot_size_distribution,
    plot_velocity_distribution,
    plot_velocity_diameter_heatmap
)


def process_event(
    event_id_int: int,
    df_event: pd.DataFrame,
    dirs: Dict[str, Path],
    diameters: List[float],
    velocities: List[float],
    combined_matrices_dir: Path,
    accumulation_intervals: List[int]
) -> Tuple[int, bool]:
    """
    Processes a single precipitation event by generating relevant plots.
    """
    try:
        # Drop NaN 'Intensidad'
        df_event = df_event.dropna(subset=['Intensidad'])
        if df_event.empty:
            logging.warning(f"All 'Intensidad' values are NaN for Event {event_id_int}. Skipping plots.")
            return (event_id_int, False)

        # Plot precipitation intensity
        plot_precipitation_intensity_separate(
            df_event=df_event,
            intervals=accumulation_intervals,
            event_id=event_id_int,
            save_dir=dirs['intensity_dir']
        )

        # Plot hyetograph
        plot_hyetograph(
            df_event=df_event,
            event_id=event_id_int,
            save_dir=dirs['hyetograph_dir']
        )

        # Plot accumulated precipitation
        plot_accumulated_precipitation(
            df_event=df_event,
            event_id=event_id_int,
            save_dir=dirs['intensity_dir']
        )

         # Load combined matrix
        combined_matrix_path = combined_matrices_dir / f'combined_event_{event_id_int}.npy'
        
        if not combined_matrix_path.exists():
            print(combined_matrices_dir)
            logging.warning(f"Combined matrix not found for Event {event_id_int}: {combined_matrix_path}")
            return (event_id_int, False)
        combined_matrix = np.load(combined_matrix_path, allow_pickle=True)

        # Validate shape
        expected_shape = (len(velocities), len(diameters))
        if combined_matrix.shape != expected_shape:
            logging.warning(f"Matrix shape {combined_matrix.shape} != expected {expected_shape} for Event {event_id_int}")
            return (event_id_int, False)

        # Plot size distribution
        plot_size_distribution(
            combined_matrix=combined_matrix,
            diameters=diameters,
            event_id=event_id_int,
            save_dir=dirs['size_dir']
        )

        # Plot velocity distribution
        plot_velocity_distribution(
            combined_matrix=combined_matrix,
            velocities=velocities,
            event_id=event_id_int,
            save_dir=dirs['velocity_dir']
        )

        # Plot velocity-diameter heatmap
        plot_velocity_diameter_heatmap(
            combined_matrix=combined_matrix,
            velocities=velocities,
            diameters=diameters,
            event_id=event_id_int,
            save_dir=dirs['heatmap_dir']
        )

        return (event_id_int, True)
    except Exception as e:
        logging.error(f"Error processing Event {event_id_int}: {e}", exc_info=True)
        return (event_id_int, False)

def visualization_main(
    start_date: str = None,
    end_date: str = None,
    config_path: str = "config/config.yaml"
):
    """
    Loop over each location’s annotated CSV and generate plots.
    """
    # 1. Early console logging
    setup_initial_logging()

    # 2. Load configuration and set up logging
    config = load_config(config_path)
    viz_cfg = config.get('visualization', {})
    log_file = viz_cfg.get('log_file_path', project_root / 'logs' / 'visualization.log')
    ensure_directory_exists(Path(log_file).parent)
    configure_logging(str(log_file))
    logging.info("Visualization Workflow Started.")

    # 3. Resolve key directories relative to project root
    # Base directory containing per-location subfolders with annotated_data.csv
    annotated_input_dir_cfg = viz_cfg.get('annotated_input_dir', 'data/processed')
    annotated_base = (project_root / annotated_input_dir_cfg).resolve()

    # Directory where combined matrices were saved per-location
    events_base = (project_root / viz_cfg.get('event_directory',
                                          'data/processed/events')).resolve()
    
    
    plots_root_cfg = viz_cfg.get('plots_output_dir', 'plots')
    plots_root = (project_root / plots_root_cfg).resolve()

    # 4. Load diameter-velocity mapping

    mapping_cfg = viz_cfg.get('diam_vel_mapping_file', 'diam_vel_mapping.csv')
    diam_vel_mapping_file = (project_root / mapping_cfg).resolve()
    logging.debug(f"Resolved diam-vel mapping file to: {diam_vel_mapping_file}")


    diameters, velocities = load_diam_vel_mapping_csv(str(diam_vel_mapping_file))

    # 5. Set accumulation intervals Set accumulation intervals
    accumulation_intervals = [1, 5, 10, 15, 30, 60]

    # 6. Iterate per-location subfolder
    for loc_dir in annotated_base.iterdir():
        if not loc_dir.is_dir():
            continue
        location = loc_dir.name
        annotated_csv = loc_dir / 'annotated_data.csv'
        combined_matrices_dir = events_base / location
        if not annotated_csv.exists():
            logging.warning(f"Annotated CSV not found for {location}")
            continue

        df_annot = pd.read_csv(annotated_csv, parse_dates=['Datetime'])
        if 'Precip_Event' not in df_annot.columns:
            logging.warning(f"No Precip_Event column in {location}'s annotated data")
            continue

        logging.info(f"Generating plots for location: {location}")
        plots_base = plots_root / location
        dirs_map = create_plots_output_directories(plots_base)

        unique_events = df_annot['Precip_Event'].dropna().unique()
        success = 0
        failure = 0
        with ProcessPoolExecutor() as executor:
            futures = []
            for eid in unique_events:
                eid_int = int(eid)
                df_event = df_annot[df_annot['Precip_Event'] == eid_int].copy()
                futures.append(
                    executor.submit(
                        process_event,
                        eid_int,
                        df_event,
                        dirs_map,
                        diameters,
                        velocities,
                        combined_matrices_dir,
                        accumulation_intervals
                    )
                )
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Plots {location}", unit="event"):
                eid_int, ok = fut.result()
                if ok:
                    success += 1
                else:
                    failure += 1

        logging.info(f"Finished {location}: {success} succeeded, {failure} failed.")

    logging.info("Visualization Workflow Completed.")


def _cli_entry_point():
    parser = argparse.ArgumentParser(description="Visualization for DSD events per location.")
    parser.add_argument('--config', default="config/config.yaml", help="Path to config.")
    parser.add_argument('--start-date', help="Override start date (YYYY-MM-DD).")
    parser.add_argument('--end-date', help="Override end date (YYYY-MM-DD).")
    args = parser.parse_args()
    visualization_main(args.start_date, args.end_date, args.config)

if __name__ == "__main__":
    _cli_entry_point()
