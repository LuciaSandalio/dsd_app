# modules/__init__.py

from .data_processing import (
    load_data,
    save_dataframe,
    save_matrices,
    process_file  # Added process_file here
)
from .event_identification import (
    remove_duplicates,
    ensure_continuous_timestamps,
    mark_precipitation_activity,
    identify_precipitation_events,
    save_annotated_data,
    extract_and_save_events
)
from .visualization import (
    resource_path,
    create_plots_output_directories,
    compute_bin_edges,
    plot_precipitation_intensity_separate,
    plot_hyetograph,
    plot_accumulated_precipitation,
    plot_size_distribution,
    plot_velocity_distribution,
    plot_velocity_diameter_heatmap
)
from .utils import (
    setup_initial_logging,    
    configure_logging,
    load_config,
    ensure_directory_exists
)

__all__ = [
    "setup_initial_logging"
    "load_config",
    "load_data",
    "save_dataframe",
    "save_matrices",
    "process_file",  
    "remove_duplicates",
    "ensure_continuous_timestamps",
    "mark_precipitation_activity",
    "identify_precipitation_events",
    "check_events_in_range",
    "compute_bin_edges",
    "plot_precipitation_intensity_separate",
    "plot_hyetograph",
    "plot_accumulated_precipitation",
    "plot_size_distribution",
    "plot_velocity_distribution",
    "plot_velocity_diameter_heatmap",
    "configure_logging",
    "ensure_directory_exists",
    "save_annotated_data",
    "extract_and_save_events",
    "resource_path",
    "create_plots_output_directories"
]

