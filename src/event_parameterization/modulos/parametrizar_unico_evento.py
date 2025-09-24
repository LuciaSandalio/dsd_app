import numpy as np
from datetime import datetime, timedelta
import time
import os 
import sys

# Añade la carpeta principal al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pydsd_modificado as pyd
from funciones.procesamiento import *
from funciones.plot import *

# Parametros 
# Diametros en los que clasifica nuestro disdrometro (Fuente: PS Luna pag.10)
diameter_bin = np.array([
    0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 
    1.062, 1.187, 1.375, 1.625, 1.875, 2.125, 2.375, 2.750, 
    3.250, 3.750, 4.250, 4.750, 5.500, 6.500, 7.500, 8.500, 
    9.500, 11.00, 13.00, 15.00, 17.00, 19.00, 21.50, 24.50,
])

# Ancho de cada bin de clasificacion (Fuente: PS Luna pag.10)
spread_bin = np.array([
    0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 
    0.125, 0.125, 0.250, 0.250, 0.250, 0.250, 0.250, 0.500, 
    0.500, 0.500, 0.500, 0.500, 1.000, 1.000, 1.000, 1.000, 
    1.000, 2.000, 2.000, 2.000, 2.000, 2.000, 3.000, 3.000,
])

velocity_bin = np.array([
    0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 
    0.85, 0.95, 1.10, 1.30, 1.50, 1.70, 1.90, 2.20, 
    2.60, 3.00, 3.40, 3.80, 4.40, 5.20, 6.00, 6.80, 
    7.60, 8.80, 10.4, 12.0, 13.6, 15.2, 17.6, 20.8,
])

vel_terminal_gk = np.array([
    0.00000000000, 0.00000000000, 0.97107483134, 1.96094149975, 
    2.70002505068, 3.29005068300, 3.78116781524, 4.20182498077, 
    4.56972620985, 4.89664159880, 5.32858521665, 5.81937341199, 
    6.23978938052, 6.60750617820, 6.93427597159, 7.36498231842, 
    7.85577051375, 8.27618648228, 8.64390327997, 8.97067307336, 
    9.40137942019, 9.89216761552, 10.3125835841, 10.6803003817, 
    11.0070701751, 11.4377765219, 11.9285647173, 12.3489806858, 
    12.7166974835, 13.0434672769, 13.4066327181, 13.79038175205
]) # Fuente: Programa de Guida


# Largo y ancho del haz de laser del disdrometro (Fuente: # Fuentes https://www.ott.com/es-la/productos/meteorologia-80/ott-parsivel2-272/)
L = 180 # longitud del laser [mm] 
W = 30  # ancho del laser [mm]
t = 60  # tiempo de sampling (60 segundos en nuestro disdrometro)

effective_sampling_area = L * (W - diameter_bin/2) # en [mm^2]

#--------------------------------------------------------------------------------------------------------------#

def main(ruta=None):
    """
    Procesar un evento de precipitación.
    
    Args:
        ruta (str): Ruta al archivo .txt combinado del evento.
                    Si es None, pedirá seleccionar uno manualmente.
    """
    print("=== Parametrizar un evento de precipitación ===")

    # Si no recibe ruta, pedirla manualmente (modo antiguo)
    if not ruta:
        ruta = seleccionar_archivo()

    if not ruta or not os.path.isfile(ruta):
        print("❌ No se seleccionó un archivo válido. Abortando.")
        return None
    
    # Formatear txt a formato PyDSD
    ruta_txt_formateado = formatear_txt_pydsd(
        ruta_txt_entrada=ruta, 
        filtrar_calidad=True
    )
    
    print(f"✅ Se inició el proceso de creación del archivo: {ruta_txt_formateado}")
    
    # Bucle de espera para asegurar que el archivo existe.
    intentos = 0
    max_intentos = 100
    espera_segundos = 3
    
    while not os.path.exists(ruta_txt_formateado) and intentos < max_intentos:
        print(f"⏳ Esperando que se cree el archivo... Intento {intentos + 1}/{max_intentos}")
        time.sleep(espera_segundos)
        intentos += 1
    
    # Comprueba si el archivo se creó.
    if os.path.exists(ruta_txt_formateado):
        print("✅ Archivo creado y encontrado. Continuamos con el código.")
        
    else:
        print("❌ Error: El archivo no se pudo crear después de varios intentos. Terminando la ejecución.")
        # Manejo del error o salida del programa.
    print(f"✅ Archivo formateado guardado en: {ruta_txt_formateado}")
    
    # Ruta manual para pruebas (eliminar cuando se integre en otro codigo)
    #ruta_txt_formateado = "C:/Users/Augusto/Desktop/DISDROMETRO/programa_pydsd/eventos_prueba/formateado_20250323_0300_60s_d3.txt" 

    # Calcular parametros 
    dsd_evento = calcular_parametros_microfisicos(
        ruta_txt_formateado=ruta_txt_formateado, 
        vel_terminal=vel_terminal_gk,
        area_efectiva_muestreo=effective_sampling_area,
        dt_medicion=60,
    )
    
    # Minutos de duracion del evento (NO contempla los datos calidad 2 ya filtrados)
    timestamps_evento = dsd_evento.time["data"]
    minutos_evento = (np.array(timestamps_evento)).shape[0]
    
    # Sacar primer timestamp del evento para nombrar archivos (Nd, Dm, D0, Dmax, Nw, mu, LWC, R)
    timestamps_raw = dsd_evento.time["data"]
    timestamps = pd.to_datetime(timestamps_raw)
    fecha_inicio = str(timestamps[0].strftime("%Y-%m-%d %H-%M"))
    
    # Graficar parametros variabilidad temporal
    graficar_parametros_dsd_con_banda(
        dsd=dsd_evento,
        ruta_txt=ruta_txt_formateado,
        ventana=5,
    )

    # Calcular parametros generales del evento para el grafico DSD
    Nd, Dm, Nw, mu = calcular_parametros_dsd(
        ruta_txt_formateado=ruta_txt_formateado,
        vel_terminal=vel_terminal_gk,
        area_efectiva_muestreo=effective_sampling_area,
        duracion_evento=minutos_evento,
        dt_medicion=60
    )

    # Extraer el primer timestamp del evento para nombrar archivos
    timestamps_raw = dsd_evento.time["data"]
    timestamps = pd.to_datetime(timestamps_raw)
    fecha_inicio = str(timestamps[0].strftime("%Y-%m-%d %H-%M"))

    # Graficar la curva DSD con el histograma de N(Di)
    graficar_DSD(
        Nw=Nw,
        Dm=Dm,
        mu=mu,
        D_bins=diameter_bin,
        Nd_bins=Nd[0],
        ruta_txt=ruta_txt_formateado,
        timestamp_str=fecha_inicio
    )
    
    graficar_violines_individuales_dsd(
        dsd=dsd_evento, 
        ruta_txt=ruta_txt_formateado,
        timestamp_str=fecha_inicio
        )

if __name__ == "__main__":
    main()




