import os
import pandas as pd
from pathlib import Path

def parametrizar_unico_evento_bucle(events_dir, raw_dir, custom_func, output_dir="data/processed/custom_events"):
    """
    Para cada evento identificado, combina los txt originales en un único archivo
    y llama a una función determinada de procesamiento (procesar_unico_evento.py)

    Args:
        events_dir (str): Carpeta donde están los event_N.csv.
        raw_dir (str): Carpeta donde están los .txt crudos.
        custom_func (callable): Función a ejecutar por cada evento. Recibe la ruta del txt combinado.
        output_dir (str): Carpeta donde guardar los txt combinados.

    Returns:
        dict: {evento: ruta_salida}
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for event_file in Path(events_dir).glob("event_*.csv"):
        df = pd.read_csv(event_file, parse_dates=["Datetime"])
        event_name = event_file.stem  # ej: "event_1"

        # Determinar qué txt usar (una hora = un archivo)
        df["Rounded_Hour"] = df["Datetime"].dt.floor("H")
        txt_files = df["Rounded_Hour"].dt.strftime("%Y%m%d_%H00.txt").unique().tolist()

        combined_file = Path(output_dir) / f"{event_name}_combined.txt"

        # Combinar los txt en un solo archivo
        with open(combined_file, "w", encoding="utf-8") as outfile:
            for fname in txt_files:
                raw_path = Path(raw_dir) / fname
                if raw_path.exists():
                    with open(raw_path, "r", encoding="utf-8") as infile:
                        outfile.writelines(infile.readlines())

        # Llamar a tu función custom
        custom_func(str(combined_file))

        results[event_name] = str(combined_file)

    return results
