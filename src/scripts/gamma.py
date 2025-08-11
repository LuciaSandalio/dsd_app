#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gamma_per_event.py

Calcula los parámetros de la curva Gamma (N0, μ, Λ) para cada evento
utilizando las matrices combinadas `combined_event_<n>.npy`.
Guarda un CSV resumen con un set de parámetros por evento.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from scipy.special import gamma, gammaln
from scipy.optimize import minimize
import pandas as pd

# --- Asegurar que 'src' esté en PYTHONPATH para importar modules.visualization ---
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent  # dsd_app/src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.visualization import load_diam_vel_mapping_csv, compute_bin_edges

# --- Configuración de logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Parámetros instrumentales: AJUSTA según tu disdrómetro ---
A  = 0.0054   # área de muestreo en m² (ej. 54 cm² → 0.0054 m²)
dt = 60.0     # tiempo de integración en s (p.ej. 60 s)

def gamma_moments(D: np.ndarray, N: np.ndarray, dD: np.ndarray) -> Tuple[float,float,float]:
    """Estimación de (N0, μ, Λ) por el Método de los Momentos."""
    mk = lambda k: np.sum(N * D**k * dD)
    M0, M3, M6 = mk(0), mk(3), mk(6)
    R = (M0 * M6) / (M3**2)
    mu  = (R - 1) / (4 - R)
    lam = ((4+mu)*(5+mu)*M3) / ((6+mu)*M6)
    N0  = M0 * lam**(mu+1) / gamma(mu+1)
    return N0, mu, lam

def neg_loglik(params: np.ndarray, D_obs: np.ndarray) -> float:
    """Negativa de la log-verosimilitud para observaciones D_obs."""
    mu, lam = params
    n = len(D_obs)
    return -(
        (mu * np.log(D_obs) - lam * D_obs).sum()
        + n * ((mu+1) * np.log(lam) - gammaln(mu+1))
    )

def mle_fit(D: np.ndarray, counts: np.ndarray, mu0: float, lam0: float
           ) -> Tuple[Union[float,None],float,float]:
    """
    Ajuste por MLE usando counts en bins D.
    Si no converge, retorna (None, mu0, lam0) indicando fallback a MoM.
    """
    D_obs = np.repeat(D, counts.astype(int))
    res = minimize(
        neg_loglik,
        x0=[mu0, lam0],
        args=(D_obs,),
        bounds=[(-0.9, None), (1e-8, None)],
        method='L-BFGS-B'
    )
    if not res.success:
        logging.warning(f"MLE no convergió: {res.message} → fallback a MoM")
        return None, mu0, lam0
    mu_mle, lam_mle = res.x
    N0_mle = len(D_obs) * lam_mle**(mu_mle+1) / gamma(mu_mle+1)
    return N0_mle, mu_mle, lam_mle

def main():
    # Rutas desde la raíz del proyecto (dsd_app)
    ROOT = SCRIPT_DIR.parent.parent  # dsd_app
    mapping_path = ROOT / "diam_vel_mapping.csv"
    events_dir   = ROOT / "data" / "processed" / "events" / "bosque_alegre" / "combined_npy"

    # 1) Cargo bins y anchos
    diameters, _ = load_diam_vel_mapping_csv(mapping_path)
    diameters = np.array(diameters)
    edges     = compute_bin_edges(diameters)
    widths    = np.diff(edges)

    # 2) Itero sobre archivos de evento
    results = []
    for npy_file in sorted(events_dir.glob("combined_event_*.npy")):
        event_str = npy_file.stem.split("combined_event_")[1]
        try:
            event_id = int(float(event_str))
        except ValueError:
            event_id = event_str  # queda como cadena si no es numérico

        mat    = np.load(npy_file)
        counts = mat.sum(axis=0)

        # filtro bins inválidos
        mask = (counts > 0) & (widths > 0)
        D      = diameters[mask]
        dD     = widths[mask]
        cnt    = counts[mask]
        if cnt.sum() == 0:
            logging.warning(f"Evento {event_id}: sin gotas válidas → salto")
            continue

        # concentración
        Nconc = cnt / (A * dt * dD)

        # MoM
        N0_mom, mu_mom, lam_mom = gamma_moments(D, Nconc, dD)

        # MLE
        N0_mle, mu_mle, lam_mle = mle_fit(D, cnt, mu_mom, lam_mom)
        if N0_mle is None:
            N0_mle = N0_mom

        results.append({
            "event_id":   event_id,
            "N0_MoM":     N0_mom,
            "mu_MoM":     mu_mom,
            "Lambda_MoM": lam_mom,
            "N0_MLE":     N0_mle,
            "mu_MLE":     mu_mle,
            "Lambda_MLE": lam_mle
        })
        logging.info(f"Evento {event_id}: MoM μ={mu_mom:.2f}, Λ={lam_mom:.2f} │ MLE μ={mu_mle:.2f}, Λ={lam_mle:.2f}")

    # 3) Guardar CSV
    if not results:
        logging.error("No se encontró ningún evento para procesar.")
        return

    df_res = pd.DataFrame(results).sort_values("event_id")
    out_csv = ROOT / "dsd_parameters_per_event.csv"
    df_res.to_csv(out_csv, index=False, float_format="%.6f")
    logging.info(f"Guardado parámetros por evento en '{out_csv}'")

if __name__ == "__main__":
    main()
