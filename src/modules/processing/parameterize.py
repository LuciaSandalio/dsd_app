
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from scipy.special import gamma
from scipy.optimize import brentq

from src.modules.processing.file_index import FileIndex

# --- Parsivel Constants (Extracted from pydsd_modificado) ---
DIAMETER_BIN_CENTERS = np.array([
    0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 
    1.062, 1.187, 1.375, 1.625, 1.875, 2.125, 2.375, 2.75, 3.25, 
    3.750, 4.250, 4.750, 5.500, 6.500, 7.500, 8.500, 9.500, 
    11.000, 13.000, 15.000, 17.000, 19.000, 21.500, 24.500
])

DIAMETER_BIN_SPREAD = np.array([
    0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
    0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5,
    1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3
])

VELOCITY_BIN_CENTERS = np.array([
    0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
    1.1, 1.3, 1.5, 1.7, 1.9, 2.2, 2.6, 3, 3.4, 3.8,
    4.4, 5.2, 6.0, 6.8, 7.6, 8.8, 10.4, 12.0, 13.6, 15.2, 17.6, 20.8
])

# Parsivel Conditional Matrix (PCM) filter from legacy code (Casanovas criterion)
# Filters out physically impossible drop velocity/diameter pairs (noise).
PCM_MATRIX_FLAT = np.array([
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
])
PCM_MATRIX = PCM_MATRIX_FLAT.reshape(32, 32)

def calculate_sampling_area(diameter_mm: np.ndarray) -> np.ndarray:
    """Effective sampling area in mm^2 (180 * (30 - 0.5D))."""
    return 180 * (30 - 0.5 * diameter_mm)

# Gunn-Kinzer Terminal Velocities (from legacy code)
VEL_TERMINAL_GK = np.array([
    0.00000000000, 0.00000000000, 0.97107483134, 1.96094149975, 
    2.70002505068, 3.29005068300, 3.78116781524, 4.20182498077, 
    4.56972620985, 4.89664159880, 5.32858521665, 5.81937341199, 
    6.23978938052, 6.60750617820, 6.93427597159, 7.36498231842, 
    7.85577051375, 8.27618648228, 8.64390327997, 8.97067307336, 
    9.40137942019, 9.89216761552, 10.3125835841, 10.6803003817, 
    11.0070701751, 11.4377765219, 11.9285647173, 12.3489806858, 
    12.7166974835, 13.0434672769, 13.4066327181, 13.79038175205
])

def _estimate_mu(m2: float, m4: float, m6: float) -> float:
    """
    Estimate the Gamma DSD shape parameter (Mu) using the Method of Moments.
    
    Uses the dimensionless ratio eta = M4^2 / (M2 * M6).
    For a Gamma DSD, eta = Gamma(mu+5)^2 / (Gamma(mu+3) * Gamma(mu+7)).
    We solve for mu numerically using Brent's method.
    
    Reference: Zhang et al. (2003), Cao & Zhang (2009)
    
    Returns:
        Estimated mu clamped to [-1, 20]. Returns 0.0 on failure.
    """
    MU_MIN = -1.0
    MU_MAX = 20.0
    
    eta_obs = (m4 ** 2) / (m2 * m6)
    
    def eta_func(mu):
        """Theoretical eta as a function of mu."""
        return (gamma(mu + 5) ** 2) / (gamma(mu + 3) * gamma(mu + 7)) - eta_obs
    
    try:
        # Check if root exists in the bracket
        fa = eta_func(MU_MIN)
        fb = eta_func(MU_MAX)
        
        if fa * fb > 0:
            # No sign change => no root in bracket, fall back to exponential
            return 0.0
        
        mu_est = brentq(eta_func, MU_MIN, MU_MAX, xtol=1e-4, maxiter=100)
        return float(np.clip(mu_est, MU_MIN, MU_MAX))
    except (ValueError, OverflowError, RuntimeError):
        return 0.0


def calculate_dsd_params(matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate DSD parameters from 32x32 raw matrix (Velocity x Diameter).
    
    Logic ported from pydsd_modificado:
    1. Calculate Nd(D) (Concentration per mm)
    2. Calculate Moments (M0, M3, M4, M6)
    3. Calculate Parameters (Nt, LWC, Z, Dm, Nw, N0, Lambda)
    """
    delta_t = 60.0  # seconds
    
    # Apply Parsivel Conditional Matrix Filter
    # Filters out noise (drops with impossible Diameter/Velocity combinations)
    matrix_filtered = matrix * PCM_MATRIX
    
    # 1. Calculate Nd (Columnwise Sum)
    # matrix shape: (32, 32) -> Rows=Velocity, Cols=Diameter
    sum_nij = np.sum(matrix_filtered, axis=0)  # Shape (32,)
    
    # Effective Area
    A_eff = calculate_sampling_area(DIAMETER_BIN_CENTERS)
    
    # Use Gunn-Kinzer Terminal Velocity (Legacy Consistency)
    v_terminal = VEL_TERMINAL_GK.copy()
    
    # Nd Calculation [m^-3 mm^-1]
    # Avoid div by zero
    v_safe = v_terminal.copy()
    v_safe[v_safe == 0] = 0.1 
    
    Nd = (1e6 * sum_nij) / (A_eff * DIAMETER_BIN_SPREAD * delta_t * v_safe)
    Nd = np.nan_to_num(Nd, nan=0.0)

    # 2. Moments Calculation
    def calc_moment(n):
        return np.sum(Nd * (DIAMETER_BIN_CENTERS ** n) * DIAMETER_BIN_SPREAD)

    m0 = calc_moment(0)
    m2 = calc_moment(2)
    m3 = calc_moment(3)
    m4 = calc_moment(4)
    m6 = calc_moment(6)
    
    # 3. Parameters
    
    # Nt [m^-3]
    Nt = m0
    
    # LWC [g/m^3]
    # Nd units: [m^-3 mm^-1], D in mm, dD in mm
    # M3 = sum(Nd * D^3 * dD) -> [m^-3 mm^3]
    # Volume per drop = (pi/6) * D^3 [mm^3]
    # Water density = 1 g/cm^3 = 1e-3 g/mm^3
    # LWC = (pi/6) * rho_w * M3 [g/m^3]
    LWC = (np.pi / 6.0) * 1e-3 * m3
    
    # Z_calc [dBZ]
    # Z = M6 [mm^6/m^3]
    Z_lin = m6
    Z_calc = 10 * np.log10(Z_lin) if Z_lin > 0 else 0.0
    
    # Dm [mm] (Mass-weighted mean diameter)
    Dm = m4 / m3 if m3 > 0 else 0.0
    
    # Nw [mm^-1 m^-3]
    if Dm > 0:
        Nw = (256.0 / (np.pi * 1e-3)) * (LWC / (Dm ** 4))
    else:
        Nw = 0.0
        
    # 4. Gamma DSD Fit: Mu, Lambda, N0
    # N(D) = N0 * D^mu * exp(-Lambda * D)
    # Using Method of Moments with eta = M4^2 / (M2 * M6)
    # Reference: Zhang et al. (2003), Cao & Zhang (2009)
    
    Mu = 0.0
    Lambda = 0.0
    N0 = 0.0
    
    if m2 > 0 and m4 > 0 and m6 > 0:
        Mu = _estimate_mu(m2, m4, m6)
        
        # Lambda from Gamma MoM (moments 2 and 4):
        # Lambda = sqrt( (M2 * Gamma(mu+5)) / (M4 * Gamma(mu+3)) )
        try:
            g_mu3 = gamma(Mu + 3)
            g_mu5 = gamma(Mu + 5)
            ratio = (m2 * g_mu5) / (m4 * g_mu3)
            if ratio > 0:
                Lambda = np.sqrt(ratio)
            
            # N0 = M2 * Lambda^(mu+3) / Gamma(mu+3)
            if Lambda > 0 and g_mu3 > 0:
                N0 = m2 * (Lambda ** (Mu + 3)) / g_mu3
        except (OverflowError, ValueError):
            # Fallback to exponential if Gamma computation overflows
            Mu = 0.0
            Lambda = 0.0
            N0 = 0.0
    
    return {
        "Nt": float(Nt),
        "N0": float(N0),
        "Lambda": float(Lambda),
        "Mu": float(Mu),
        "LWC": float(LWC),
        "Z_calc": float(Z_calc),
        "Dm": float(Dm),
        "Nw": float(Nw)
    }

def parameterize_event(event_csv: Path, output_file: Path, manifest_file: Path, file_index: FileIndex):
    """
    Process a single event CSV:
    1. Read event data.
    2. For each minute, load 32x32 matrix.
    3. Calculate DSD params.
    4. Save result.
    """
    df = pd.read_csv(event_csv)
    
    results = []
    manifest_entries = []
    
    # Required columns check
    if "Intensidad" not in df.columns:
        logging.warning(f"Skipping {event_csv.name}: Missing 'Intensidad'")
        return

    for _, row in df.iterrows():
        # Use the original Timestamp column directly (already epoch seconds)
        # Avoids timezone re-conversion issues with pd.to_datetime().timestamp()
        try:
            ts_int = int(row['Timestamp'])
        except (KeyError, ValueError, TypeError):
            logging.warning(f"Invalid timestamp in {event_csv.name}: row {_}")
            continue
            
        matrix_path = file_index.matrix_map.get(ts_int)
        
        params = {
            "Nt": 0.0, "N0": 0.0, "Lambda": 0.0, "Mu": 0.0, 
            "LWC": 0.0, "Z_calc": 0.0, "Dm": 0.0, "Nw": 0.0
        }
        
        status = "missing_matrix"
        
        # Minimum particle count filter: skip DSD computation if too few drops
        MIN_PARTICLES = 10
        try:
            particle_count = int(row.get('Cant_Particulas', 0))
        except (ValueError, TypeError):
            particle_count = 0
        
        if particle_count < MIN_PARTICLES:
            status = "low_particle_count"
        elif matrix_path and matrix_path.exists():
            try:
                matrix = np.load(matrix_path)
                params = calculate_dsd_params(matrix)
                status = "computed"
            except Exception as e:
                logging.error(f"Error calcing params for {ts_int}: {e}")
                status = "error"
        
        # Merge results
        entry = row.to_dict()
        entry.update(params)
        entry['status'] = status
        results.append(entry)
        
        manifest_entries.append({
            "timestamp": str(ts_int),
            "matrix_source": str(matrix_path) if matrix_path else None,
            "status": status
        })
        
    # Save
    out_df = pd.DataFrame(results)
    
    # Drop redundant columns
    cols_to_drop = ['dt_obj']
    out_df.drop(columns=[c for c in cols_to_drop if c in out_df.columns], inplace=True)
    
    out_df.to_csv(output_file, index=False)
    
    with open(manifest_file, 'w') as f:
        json.dump(manifest_entries, f, indent=2)
        
    logging.info(f"Saved params to {output_file} ({len(results)} rows)")
