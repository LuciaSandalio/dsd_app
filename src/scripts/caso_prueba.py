import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy.special import gamma, gammaln
from scipy.optimize import minimize

# --- Asegurar que 'src' esté en PYTHONPATH para importar modules.visualization ---
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent  # dsd_app/src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Parámetros instrumentales
A = 0.0054   # área de muestreo (m²)
dt = 60.0    # tiempo de integración (s)

# Rutas desde la raíz del proyecto (dsd_app)
ROOT = SCRIPT_DIR.parent.parent  # dsd_app
mapping_path = ROOT / "diam_vel_mapping.csv"
matrix_path   = ROOT / "data" / "processed" / "events" / "bosque_alegre" / "combined_npy" / "combined_event_8.0.npy"


# 1) Cargar datos
mat = np.load(matrix_path)
mapping = pd.read_csv(mapping_path, decimal=',')
diameters = mapping['diameters_mm'].values

# 2) Calcular bordes y anchos de bin
edges = np.zeros(len(diameters) + 1)
edges[1:-1] = (diameters[:-1] + diameters[1:]) / 2
edges[0] = diameters[0] - (diameters[1] - diameters[0]) / 2
edges[-1] = diameters[-1] + (diameters[-1] - diameters[-2]) / 2
widths = np.diff(edges)

# 3) Sumar conteos por diámetro
counts = mat.sum(axis=0)

# 4) Filtrar bins inválidos
mask = (counts > 0) & (widths > 0)
D = diameters[mask]
dD = widths[mask]
cnt = counts[mask]

# 5) Convertir a concentración N(D)
Nconc = cnt / (A * dt * dD)

# 6) Función Método de los Momentos
def gamma_moments(D, N, dD):
    mk = lambda k: np.sum(N * D**k * dD)
    M0, M3, M6 = mk(0), mk(3), mk(6)
    R = (M0 * M6) / (M3**2)
    mu  = (R - 1) / (4 - R)
    lam = ((4 + mu) * (5 + mu) * M3) / ((6 + mu) * M6)
    N0  = M0 * lam**(mu + 1) / gamma(mu + 1)
    return N0, mu, lam

# 7) Función MLE
def neg_loglik(params, D_obs):
    mu, lam = params
    n = len(D_obs)
    return -(
        (mu * np.log(D_obs) - lam * D_obs).sum()
        + n * ((mu + 1) * np.log(lam) - gammaln(mu + 1))
    )

def mle_fit(D, counts, mu0, lam0):
    D_obs = np.repeat(D, counts.astype(int))
    res = minimize(
        neg_loglik,
        x0=[mu0, lam0],
        args=(D_obs,),
        bounds=[(-0.9, None), (1e-8, None)],
        method='L-BFGS-B'
    )
    if res.success:
        mu_mle, lam_mle = res.x
        N0_mle = len(D_obs) * lam_mle**(mu_mle + 1) / gamma(mu_mle + 1)
        return N0_mle, mu_mle, lam_mle
    else:
        raise RuntimeError("MLE no convergió: " + res.message)

# 8) Cálculo de parámetros
N0_mom, mu_mom, lam_mom = gamma_moments(D, Nconc, dD)
N0_mle, mu_mle, lam_mle = mle_fit(D, cnt, mu_mom, lam_mom)

# 9) Mostrar resultados
results = pd.DataFrame({
    "Método": ["MoM", "MLE"],
    "N0":      [N0_mom,      N0_mle],
    "mu":      [mu_mom,      mu_mle],
    "Lambda":  [lam_mom,     lam_mle]
})
print("Parámetros de la curva Gamma para el evento 3.0:\n")
print(results.to_string(index=False))

# 10) Graficar ajuste
D_plot = np.linspace(D.min(), D.max(), 200)
fit_mom = N0_mom * D_plot**mu_mom * np.exp(-lam_mom * D_plot)
fit_mle = N0_mle * D_plot**mu_mle * np.exp(-lam_mle * D_plot)

plt.figure(figsize=(6,4))
plt.loglog(D,   Nconc,    'o', label="Observado", markersize=5)
#plt.loglog(D_plot, fit_mom, '-', label="Gamma MoM")
plt.loglog(D_plot, fit_mle, '--', label="Gamma MLE")
plt.xlabel("Diámetro D (mm)")
plt.ylabel("N(D) (gotas·m⁻³·mm⁻¹)")
plt.title("Ajuste DSD - Evento 2025-03-07 23:59 - 2025-03-08 10:43")
plt.legend()
plt.tight_layout()
plt.show()
