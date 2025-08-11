#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Leer y convertir columnas numéricas
df = pd.read_csv("/home/chula/Documentos/python/dsd_app/dsd_parameters_per_event.csv")

# Forzar columnas a float
num_cols = ["N0_MoM","mu_MoM","Lambda_MoM","N0_MLE","mu_MLE","Lambda_MLE"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2. Dominio de diámetros para las curvas
D = np.linspace(0.1, 10.0, 200)

# 3. Graficar por evento
for _, row in df.iterrows():
    event_id = int(row["event_id"])

    # Parámetros como floats
    N0_mom, mu_mom, lam_mom = row["N0_MoM"],    row["mu_MoM"],    row["Lambda_MoM"]
    N0_mle, mu_mle, lam_mle = row["N0_MLE"],    row["mu_MLE"],    row["Lambda_MLE"]

    # Generar curvas
    N_mom = N0_mom * D**mu_mom * np.exp(-lam_mom * D)
    N_mle = N0_mle * D**mu_mle * np.exp(-lam_mle * D)

    # Plot log-log
    plt.figure()
    plt.loglog(D, N_mom, label="Gamma MoM", linewidth=2)
    plt.loglog(D, N_mle, label="Gamma MLE", linewidth=2)
    plt.xlabel("Diámetro D (mm)")
    plt.ylabel("N(D)")
    plt.title(f"Evento {event_id}")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)

    # Guardar figura
    out_png = f"gamma_event_{event_id}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Guardado → {out_png}")
