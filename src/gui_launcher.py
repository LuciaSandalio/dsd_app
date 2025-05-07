#../src/gui_launcher.py

import tkinter as tk
from tkinter import messagebox
import sys
from pathlib import Path
from scripts.master_workflow import run_master_workflow

def run_workflow():
    start_date = start_entry.get().strip()
    end_date = end_entry.get().strip()

    if not start_date or not end_date:
        messagebox.showwarning("Missing Dates", "Please provide both start and end dates.")
        return

    # Validate date format (optional)
    # If you want to strictly validate YYYY-MM-DD, you can do:
    # import datetime
    # try:
    #     datetime.datetime.strptime(start_date, "%Y-%m-%d")
    #     datetime.datetime.strptime(end_date, "%Y-%m-%d")
    # except ValueError:
    #     messagebox.showerror("Invalid Date Format", "Dates must be in YYYY-MM-DD format.")
    #     return

    # Set project_root to be the parent of src (i.e. dsd_app_installable)
    project_root = Path(__file__).parent.parent.resolve()

    command = [
        sys.executable,
        str(project_root / "src" / "scripts" / "master_workflow.py"),
        "--config",
        str(project_root / "config" / "config.yaml"),
        "--start-date",
        start_date,
        "--end-date",
        end_date
    ]

    run_button.config(state=tk.DISABLED)
    status_label.config(text="Running workflow, please wait...")
    root.update_idletasks()

    try:
        # Build the config file path
        project_root = Path(__file__).parent.parent.resolve()
        config_file = project_root / "config" / "config.yaml"

        # Call the master workflow in-process
        run_master_workflow(start_date, end_date, str(config_file))

        status_label.config(text="Workflow completed successfully. Closing...")
    except Exception as e:
        status_label.config(text="Workflow encountered an error. Check logs. Closing...")

    # Wait a moment before closing to allow user to see message
    root.update_idletasks()
    root.after(2000, root.destroy)  # Closes the GUI after 2 seconds

root = tk.Tk()
root.title("DSD Workflow")

tk.Label(root, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, padx=10, pady=5)
start_entry = tk.Entry(root)
start_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="End Date (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=5)
end_entry = tk.Entry(root)
end_entry.grid(row=1, column=1, padx=10, pady=5)

run_button = tk.Button(root, text="Run Workflow", command=run_workflow)
run_button.grid(row=2, column=0, columnspan=2, pady=10)

status_label = tk.Label(root, text="")
status_label.grid(row=3, column=0, columnspan=2)

root.mainloop()
