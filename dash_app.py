# app.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import threading
import subprocess
import pandas as pd
from pathlib import Path
import logging
import sys
import base64
from typing import Optional, Tuple
from datetime import datetime

from modules.utils import configure_logging


# Initialize the Dash app with Bootstrap for better styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment

# Configure logging for the Dash app using centralized function
configure_logging('logs/dash_app.log')

# Define global variables to track workflow status
workflow_lock = threading.Lock()
workflow_completed = False
workflow_success = False
workflow_message = ""

def run_workflow_thread(start_date: str, end_date: str, config_path: str = 'config/config.yaml'):
    global workflow_completed, workflow_success, workflow_message
    try:
        logging.info(f"Workflow started for dates: {start_date} to {end_date}")
        # Run master_workflow.py as a subprocess
        command = [sys.executable, 'scripts/master_workflow.py', '--config', config_path,
                   '--start-date', start_date, '--end-date', end_date]
        subprocess.run(command, check=True)
        logging.info("Master workflow completed successfully.")

        with workflow_lock:
            workflow_completed = True
            workflow_success = True
            workflow_message = "Workflow completed successfully."
    except subprocess.CalledProcessError as e:
        logging.error(f"Workflow failed: {e}")
        with workflow_lock:
            workflow_completed = True
            workflow_success = False
            workflow_message = "Workflow failed. Check logs for details."
    except Exception as e:
        logging.error(f"Unexpected error during workflow: {e}")
        with workflow_lock:
            workflow_completed = True
            workflow_success = False
            workflow_message = "Workflow encountered an unexpected error."

# Function to safely read the CSV with date parsing
def get_date_range(csv_path: str) -> Tuple[Optional[Path], Optional[datetime], Optional[datetime]]:
    """
    Reads the CSV file and extracts the minimum and maximum dates from the 'Datetime' column.

    Parameters:
    - csv_path (str): Path to the CSV file.

    Returns:
    - Tuple containing:
        - Path object if file exists, else None
        - Minimum date as datetime object if available, else None
        - Maximum date as datetime object if available, else None
    """
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        logging.warning(f"CSV file does not exist: {csv_path}")
        return None, None, None

    try:
        df = pd.read_csv(csv_path, parse_dates=['Datetime'], dayfirst=True)
        min_date = df['Datetime'].min()
        max_date = df['Datetime'].max()
        if pd.isnull(min_date) or pd.isnull(max_date):
            logging.warning(f"No valid dates found in 'Datetime' column of {csv_path}")
            return csv_path_obj, None, None
        logging.info(f"Date range from CSV: {min_date.date()} to {max_date.date()}")
        return csv_path_obj, min_date, max_date
    except Exception as e:
        logging.error(f"Failed to read CSV file {csv_path}: {e}")
        return csv_path_obj, None, None

# Define the layout of the Dash app
def create_layout():

    min_date_allowed = datetime(2020, 1, 1).date()
    max_date_allowed = datetime.today().date()
    start_date = min_date_allowed
    end_date = max_date_allowed

    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Precipitation Data Dashboard"), width={'size': 6, 'offset': 3},
                    className='text-center mb-4')
        ]),

        dbc.Row([
            dbc.Col([
                html.Label("Select Date Range:"),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    min_date_allowed=min_date_allowed,
                    max_date_allowed=max_date_allowed,
                    start_date=start_date,
                    end_date=end_date,
                    display_format='YYYY-MM-DD',
                    style={'margin': '0 auto'}
                ),
            ], width={'size': 6, 'offset': 3}, className='text-center mb-3')
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Button("Run Workflow", id='run-workflow-button', color='primary', className='mr-2'),
            ], width={'size': 6, 'offset': 3}, className='text-center mb-4')
        ]),

        dbc.Row([
            dbc.Col([
                html.Div(id='workflow-status', className='text-center')
            ], width={'size': 6, 'offset': 3}, className='mb-4')
        ]),

        dbc.Row([
            dbc.Col([
                html.Div(id='summary-stats', className='text-center')
            ], width={'size': 8, 'offset': 2}, className='mb-4')
        ]),

        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-intensity",
                    type="default",
                    children=html.Img(id='intensity-plot', src='', style={'width': '100%', 'height': 'auto'})
                )
            ], width=6),
            dbc.Col([
                dcc.Loading(
                    id="loading-hyetograph",
                    type="default",
                    children=html.Img(id='hyetograph-plot', src='', style={'width': '100%', 'height': 'auto'})
                )
            ], width=6)
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-particle-distribution",
                    type="default",
                    children=html.Img(id='particle-distribution-plot', src='', style={'width': '100%', 'height': 'auto'})
                )
            ], width=6),
            dbc.Col([
                dcc.Loading(
                    id="loading-heatmap",
                    type="default",
                    children=html.Img(id='heatmap-plot', src='', style={'width': '100%', 'height': 'auto'})
                )
            ], width=6)
        ], className='mb-4'),

    ], fluid=True)


app.layout = create_layout()

# Callback to handle workflow execution and UI updates
@app.callback(
    [
        Output('workflow-status', 'children'),
        Output('run-workflow-button', 'disabled'),
        Output('summary-stats', 'children'),
        Output('intensity-plot', 'src'),
        Output('hyetograph-plot', 'src'),
        Output('particle-distribution-plot', 'src'),
        Output('heatmap-plot', 'src')
    ],
    [Input('run-workflow-button', 'n_clicks')],
    [
        State('date-picker-range', 'start_date'),
        State('date-picker-range', 'end_date')
    ]
)
def handle_workflow(n_clicks, start_date, end_date):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    if not start_date or not end_date:
        return "Please select a valid date range.", False, "", "", "", "", ""

    # Disable the button to prevent multiple clicks
    button_disabled = True
    status_message = "Workflow is running. Please wait..."

    # Start the workflow in a separate thread
    thread = threading.Thread(target=run_workflow_thread, args=(start_date, end_date))
    thread.start()

    # Wait for the thread to complete
    thread.join()

    # After workflow completion, update the UI based on success or failure
    if workflow_success:
        status_message = "Workflow completed successfully."
        # Load summary statistics
        try:
            summary_df = pd.read_csv('data/processed/summary_stats.csv', parse_dates=['Date'])  # Ensure 'Date' is parsed
            total_days = summary_df['total_days'].iloc[0]
            total_precipitation = summary_df['total_precipitation'].iloc[0]
            max_precip = summary_df['max_precipitation'].iloc[0]
            max_precip_day = summary_df['max_precipitation_day'].iloc[0]
            
            summary = html.Div([
                html.H2("Summary Statistics"),
                html.Ul([
                    html.Li(f"Total Number of Days: {total_days}"),
                    html.Li(f"Total Precipitation: {total_precipitation:.2f} mm"),
                    html.Li(f"Day with Maximum Precipitation: {max_precip_day}"),
                    html.Li(f"Maximum Precipitation Value: {max_precip:.2f} mm"),
                ], style={'listStyleType': 'none', 'fontSize': '20px'})
            ], style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'borderRadius': '10px'})
        except Exception as e:
            logging.error(f"Error loading summary statistics: {e}")
            summary = "Error loading summary statistics."

        # Load and encode plot images
        def encode_image(path):
            try:
                with open(path, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode('ascii')
                return f"data:image/png;base64,{encoded}"
            except Exception as e:
                logging.error(f"Error encoding image {path}: {e}")
                return ""

        # Define paths to the latest plots
        # Adjust the event_id as needed or implement dynamic selection
        # For demonstration, assuming event_id=1
        intensity_plot_path = Path('plots/intensity_plots/event_1_precipitation_intensity_1min.png')  # Example
        hyetograph_plot_path = Path('plots/hyetographs/event_1_hyetograph.png')  # Example
        particle_plot_path = Path('plots/size_distributions/event_1_size_distribution.png')  # Example
        heatmap_plot_path = Path('plots/heatmaps/event_1_velocity_diameter_heatmap.png')  # Example

        # Encode images
        intensity_src = encode_image(intensity_plot_path) if intensity_plot_path.exists() else ""
        hyetograph_src = encode_image(hyetograph_plot_path) if hyetograph_plot_path.exists() else ""
        particle_src = encode_image(particle_plot_path) if particle_plot_path.exists() else ""
        heatmap_src = encode_image(heatmap_plot_path) if heatmap_plot_path.exists() else ""

        return status_message, False, summary, intensity_src, hyetograph_src, particle_src, heatmap_src

    else:
        # If workflow failed
        status_message = workflow_message
        button_disabled = False
        return status_message, button_disabled, "", "", "", "", ""


if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
