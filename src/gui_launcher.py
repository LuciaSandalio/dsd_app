#!/usr/bin/env python3
# src/gui_launcher.py
# Simple GUI: write dates into config.yaml, then run master workflow (viz optional)

import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import yaml

# Path setup (no heavy imports here)
PROJ_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJ_ROOT / "config" / "config.yaml"

# Import after path resolution
import sys
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))
from scripts.master_workflow import run_master_workflow

def _write_dates_to_config(start_date: str, end_date: str) -> None:
    cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    # Your schema: top-level start_date/end_date
    cfg["start_date"] = str(start_date)
    cfg["end_date"]   = str(end_date)

    # Also reflect into events.* (optional, helpful for period combine)
    cfg.setdefault("events", {})
    cfg["events"]["start_date"] = str(start_date)
    cfg["events"]["end_date"]   = str(end_date)

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def run_workflow():
    start_date = start_entry.get().strip()
    end_date = end_entry.get().strip()
    viz_flag = viz_var.get()

    if not start_date or not end_date:
        messagebox.showwarning("Fechas faltantes", "Completá inicio y fin (YYYY-MM-DD).")
        return

    try:
        _write_dates_to_config(start_date, end_date)
    except Exception as e:
        messagebox.showerror("Config error", f"No pude actualizar config.yaml:\n{e}")
        return

    run_button.config(state=tk.DISABLED)
    status_label.config(text=f"Ejecutando workflow ({'con' if viz_flag else 'sin'} visualización)…")
    root.update_idletasks()

    try:
        # master_workflow leerá las fechas desde config.yaml
        run_master_workflow(
            config_path=str(CONFIG_PATH),
            enable_visualization=viz_flag
        )
        status_label.config(text="Workflow finalizado. Cerrando…")
    except Exception:
        status_label.config(text="Error en workflow. Revisá logs. Cerrando…")

    root.update_idletasks()
    root.after(2000, root.destroy)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("DSD Workflow")

tk.Label(root, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, padx=10, pady=5, sticky="e")
start_entry = tk.Entry(root, width=16); start_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="End Date (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=5, sticky="e")
end_entry = tk.Entry(root, width=16); end_entry.grid(row=1, column=1, padx=10, pady=5)

viz_var = tk.BooleanVar(value=False)  # visualization disabled by default
tk.Checkbutton(root, text="Run visualization stage (experimental)", variable=viz_var)\
  .grid(row=2, column=0, columnspan=2, padx=10, pady=(0,10))

run_button = tk.Button(root, text="Run Workflow", command=run_workflow, width=20)
run_button.grid(row=3, column=0, columnspan=2, pady=10)

status_label = tk.Label(root, text="", anchor="w")
status_label.grid(row=4, column=0, columnspan=2, padx=10, pady=(0,10), sticky="w")

root.mainloop()
