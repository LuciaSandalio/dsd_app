#!/usr/bin/env python3
# src/scripts/gui_launcher.py
# Modern GUI for DSD Processing Pipeline.
# Wraps master_run.py with Site Selection, Date Picker, and Live Logs.

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import sys
import subprocess
import threading
import queue

# Path setup
_THIS = Path(__file__).resolve()
PROJ_ROOT = _THIS.parents[2]
RAW_DIR = PROJ_ROOT / "data" / "raw"

class DSDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DSD Processing App")
        self.root.geometry("600x500")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- Variables ---
        self.site_var = tk.StringVar()
        self.start_date_var = tk.StringVar()
        self.end_date_var = tk.StringVar()
        self.is_running = False
        
        # --- Layout ---
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. Site Selection
        ttk.Label(main_frame, text="Select Site:").grid(row=0, column=0, sticky="w", pady=5)
        self.site_combo = ttk.Combobox(main_frame, textvariable=self.site_var, state="readonly")
        self.site_combo.grid(row=0, column=1, sticky="ew", padx=10, pady=5)
        self.refresh_sites()
        
        # 2. Date Range
        ttk.Label(main_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky="w", pady=5)
        self.start_entry = ttk.Entry(main_frame, textvariable=self.start_date_var)
        self.start_entry.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        
        ttk.Label(main_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky="w", pady=5)
        self.end_entry = ttk.Entry(main_frame, textvariable=self.end_date_var)
        self.end_entry.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
        
        # 3. Action Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        self.run_btn = ttk.Button(btn_frame, text="Run Processing", command=self.start_processing)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        
        quit_btn = ttk.Button(btn_frame, text="Quit", command=root.destroy)
        quit_btn.pack(side=tk.LEFT, padx=5)
        
        # 4. Log Output
        ttk.Label(main_frame, text="Processing Log:").grid(row=4, column=0, sticky="w")
        self.log_text = tk.Text(main_frame, height=15, width=70, state="disabled", bg="#f0f0f0")
        self.log_text.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, command=self.log_text.yview)
        scrollbar.grid(row=5, column=2, sticky='ns')
        self.log_text['yscrollcommand'] = scrollbar.set
        
        # Resize config
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)

    def refresh_sites(self):
        """Scan data/raw for sites."""
        sites = []
        if RAW_DIR.exists():
            for item in RAW_DIR.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    sites.append(item.name)
        
        # Add 'All Sites' option at the top
        combo_values = ["All Sites"] + sorted(sites)
        self.site_combo['values'] = combo_values
        self.site_combo.current(0)
            
    def log(self, message):
        """Thread-safe logging to text widget."""
        def _log():
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state="disabled")
        self.root.after(0, _log)

    def start_processing(self):
        if self.is_running: return
        
        site = self.site_var.get()
        start = self.start_date_var.get().strip()
        end = self.end_date_var.get().strip()
        
        if not site:
            messagebox.showwarning("Missing Input", "Please select a site.")
            return
            
        # Optional: Validate dates regex
        
        self.is_running = True
        self.run_btn.config(state="disabled")
        self.log_text.config(bg="white")
        
        display_site = site if site != "All Sites" else "ALL SITES"
        self.log(f"--- Starting Pipeline for {display_site} [{start} to {end}] ---")
        
        # Run in thread
        threading.Thread(target=self._run_master_script, args=(site, start, end), daemon=True).start()
        
    def _run_master_script(self, site, start, end):
        script_path = PROJ_ROOT / "src" / "scripts" / "master_run.py"
        
        cmd = [sys.executable, str(script_path)]
        
        # Only add --site if it's NOT "All Sites"
        if site != "All Sites":
            cmd += ["--site", site]
            
        if start: cmd += ["--start-date", start]
        if end: cmd += ["--end-date", end]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                self.log(line.strip())
                
            process.wait()
            
            if process.returncode == 0:
                self.log(">>> SUCCESS: Processing Completed.")
                messagebox.showinfo("Success", "Processing completed successfully!")
            else:
                self.log(f">>> FAILED (Exit Code {process.returncode})")
                messagebox.showerror("Error", "Processing failed. Check logs.")
                
        except Exception as e:
            self.log(f"Error launching subprocess: {e}")
            messagebox.showerror("Error", f"Failed to launch script: {e}")
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.run_btn.config(state="normal"))

if __name__ == "__main__":
    root = tk.Tk()
    app = DSDApp(root)
    root.mainloop()
