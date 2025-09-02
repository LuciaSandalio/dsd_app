# modules/__init__.py

# ---- Data processing ----------------------------------------------------------
from .data_processing import (
    load_data,
    save_dataframe,
    save_matrices,
    process_file,
    apply_qc,
    _normalize_matrix_size as normalize_matrix_size,  # public alias
)

# ---- Event identification -----------------------------------------------------
from .event_identification import (
    remove_duplicates,
    ensure_continuous_timestamps,
    mark_precipitation_activity,
    identify_precipitation_events,
    save_annotated_data,
    extract_and_save_events,
)

# ---- Visualization ------------------------------------------------------------
from .visualization import (
    resource_path,
    create_plots_output_directories,
    compute_bin_edges,
    plot_precipitation_intensity_separate,
    plot_hyetograph,
    plot_accumulated_precipitation,
    plot_size_distribution,
    plot_velocity_distribution,
    plot_velocity_diameter_heatmap,
)

# ---- Utilities / infrastructure ----------------------------------------------
from .utils import (
    setup_initial_logging,
    configure_logging,
    load_config,
    ensure_directory_exists,
    parse_filename_to_date,
    cleanup_from_config,
    cleanup_output,  # compatibility alias that calls cleanup_from_config
    coerce_date_str,
    iter_files_with_ext,
    filter_files_by_window,
)

__all__ = [
    # utils / infra
    "setup_initial_logging",
    "configure_logging",
    "load_config",
    "ensure_directory_exists",
    "parse_filename_to_date",
    "cleanup_from_config",
    "cleanup_output",
    "coerce_date_str",
    "iter_files_with_ext",
    "filter_files_by_window",

    # data processing
    "load_data",
    "save_dataframe",
    "save_matrices",
    "process_file",
    "apply_qc",
    "normalize_matrix_size",

    # event identification
    "remove_duplicates",
    "ensure_continuous_timestamps",
    "mark_precipitation_activity",
    "identify_precipitation_events",
    "save_annotated_data",
    "extract_and_save_events",

    # visualization
    "resource_path",
    "create_plots_output_directories",
    "load_diam_vel_mapping_from_config",
    "compute_bin_edges",
    "plot_precipitation_intensity_separate",
    "plot_hyetograph",
    "plot_accumulated_precipitation",
    "plot_size_distribution",
    "plot_velocity_distribution",
    "plot_velocity_diameter_heatmap",
]
