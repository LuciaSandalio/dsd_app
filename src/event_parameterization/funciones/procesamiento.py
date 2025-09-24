import numpy as np
from scipy.stats import mode
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import tempfile
import pydsd_modificado as pyd
from io import StringIO
import tkinter as tk
from tkinter import filedialog

#---------------------PRE-PROCESAMIENTO------------------------------------#
def seleccionar_archivo():
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal
    archivo = filedialog.askopenfilename(
        title="Seleccioná el archivo .txt del evento",
        filetypes=[("Archivos de texto", "*.txt")]
    )
    return archivo

def formatear_txt_pydsd(ruta_txt_entrada, filtrar_calidad=True):
    """
    Toma el txt con el formato recibido por el disdrometro 
    y crea un txt con el formato que recibe la libreria PyDSD en la misma carpeta.
    """
    if not os.path.isfile(ruta_txt_entrada):
        raise FileNotFoundError(f"❌ No se encontró el archivo: {ruta_txt_entrada}")

    try:
        # Descomponer el txt en variables y filtrarlas
        variables_evento = parse_file(ruta_txt_entrada)
        if not variables_evento or not variables_evento.get("timestamps"):
            raise ValueError("El archivo no contiene datos válidos.")

        variables_evento_filtradas = filtrar_por_status(filtrar_calidad, variables_evento)
        if not variables_evento_filtradas or not variables_evento_filtradas.get("timestamps"):
            raise ValueError("No quedaron registros válidos tras aplicar el filtro de calidad.")

        timestamps_convertidos = convertir_timestamps(variables_evento_filtradas["timestamps"])

        # Definir ruta del archivo formateado
        nombre_base = os.path.basename(ruta_txt_entrada)
        nombre_salida = f"formateado_{nombre_base}"
        carpeta_salida = os.path.dirname(ruta_txt_entrada)
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)

        # Escritura segura
        with open(ruta_salida, "w", encoding="latin-1") as out:
            filas_totales = len(variables_evento_filtradas["timestamps"])

            for fila in range(filas_totales):
                try:
                    fecha, hora = timestamps_convertidos[fila].split(" ")

                    out.write(f"21: {fecha}\n")
                    out.write(f"20: {hora}\n")
                    out.write(f"01: {variables_evento_filtradas['intensidades'][fila]}\n")
                    out.write(f"04: {variables_evento_filtradas['weather_codes'][fila]}\n")
                    out.write(f"07: {variables_evento_filtradas['reflectivities'][fila]}\n")
                    out.write(f"09: {variables_evento_filtradas['sample_intervals'][fila]}\n")
                    out.write(f"11: {variables_evento_filtradas['number_of_particles'][fila]}\n")
                    out.write(f"34: {variables_evento_filtradas['kinetic_energies'][fila]}\n")

                    nd_str = ";".join(f"{x}" for x in variables_evento_filtradas['nd_arrays'][fila])
                    vd_str = ";".join(f"{x}" for x in variables_evento_filtradas['vd_arrays'][fila])
                    raw_str = ";".join(f"{int(x)}" for x in variables_evento_filtradas['raw_matrices'][fila])

                    out.write(f"90: {nd_str}\n")
                    out.write(f"91: {vd_str}\n")
                    out.write(f"93: {raw_str}\n")
                    out.write("\n")
                except Exception as e:
                    print(f"⚠️ Error al procesar fila {fila}: {e}")
                    continue  # Saltar fila dañada sin romper todo

        return ruta_salida

    except Exception as e:
        raise RuntimeError(f"❌ Error en formatear_txt_pydsd: {e}")


def procesar_archivo():
    # Suponiendo que 'ruta' es la ruta de entrada para tu función.
    ruta = 'ruta/de/entrada.txt'
    
    # Llama a la función que crea el archivo.
    ruta_txt_formateado = formatear_txt_pydsd(
        ruta_txt_entrada=ruta, 
        filtrar_calidad=True
    )
    
    print(f"✅ Se inició el proceso de creación del archivo: {ruta_txt_formateado}")
    
    # Bucle de espera para asegurar que el archivo existe.
    intentos = 0
    max_intentos = 10
    espera_segundos = 1
    
    while not os.path.exists(ruta_txt_formateado) and intentos < max_intentos:
        print(f"⏳ Esperando que se cree el archivo... Intento {intentos + 1}/{max_intentos}")
        time.sleep(espera_segundos)
        intentos += 1
    
    # Comprueba si el archivo se creó.
    if os.path.exists(ruta_txt_formateado):
        print("✅ Archivo creado y encontrado. Continuamos con el código.")
        # Aquí puedes llamar a la siguiente función que usa el archivo.
        # siguiente_funcion(ruta_txt_formateado)
    else:
        print("❌ Error: El archivo no se pudo crear después de varios intentos. Terminando la ejecución.")
        # Manejo del error o salida del programa.

def parse_file(filepath):
    """
Parsea un archivo de texto con múltiples líneas de datos de disdrómetro.  
Extrae variables básicas, una matriz 32x32 (raw_matrix), y dos vectores de 32 elementos (nd y vd) por línea.

Argumentos:
- filepath (str): Ruta al archivo .txt con los datos.

Retorna:
- dict: Diccionario con los siguientes campos, cada uno como lista por línea:
    - 'timestamps' (str)
    - 'intensidades' (float)
    - 'sample_intervals' (int)
    - 'number_of_particles' (int)
    - 'weather_codes' (int)
    - 'kinetic_energies' (float)
    - 'reflectivities' (float)
    - 'status_codes' (int)
    - 'raw_matrices' (list of 1024 floats por línea)
    - 'nd_arrays' (list of 32 floats por línea)
    - 'vd_arrays' (list of 32 floats por línea)
"""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    timestamps, intensidades, sample_intervals = [], [], []
    number_of_particles, weather_codes, kinetic_energies = [], [], []
    reflectivities, status_codes, raw_matrices, nd_arrays, vd_arrays = [], [], [], [], []

    try:
        with open(filepath, 'r', encoding="utf-8", errors="ignore") as file:
            for line_num, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    timestamp_str, data_str = line.strip().split(' ', 1)
                    parts = data_str.split(";")

                    if len(parts) < (7 + 1024 + 32 + 32):
                        raise ValueError(f"Línea {line_num} incompleta: {len(parts)} elementos")

                    timestamps.append(timestamp_str)
                    intensidades.append(float(parts[0]))
                    sample_intervals.append(int(parts[1]))
                    number_of_particles.append(int(parts[2]))
                    weather_codes.append(int(parts[3]))
                    kinetic_energies.append(float(parts[4]))
                    reflectivities.append(float(parts[5]))
                    status_codes.append(int(parts[6]))

                    raw_matrices.append(list(map(float, parts[7:7+1024])))
                    nd_arrays.append(list(map(float, parts[7+1024:7+1024+32])))
                    vd_arrays.append(list(map(float, parts[7+1056:7+1056+32])))

                except Exception as e:
                    print(f"⚠️ Error parseando línea {line_num}: {e}")
                    continue  # Saltar línea problemática

        return {
            "timestamps": timestamps,
            "intensidades": intensidades,
            "sample_intervals": sample_intervals,
            "number_of_particles": number_of_particles,
            "weather_codes": weather_codes,
            "kinetic_energies": kinetic_energies,
            "reflectivities": reflectivities,
            "status_codes": status_codes,
            "raw_matrices": raw_matrices,
            "nd_arrays": nd_arrays,
            "vd_arrays": vd_arrays
        }

    except Exception as e:
        raise RuntimeError(f"❌ Error al leer {filepath}: {e}")


def filtrar_por_status(filtrar_calidad, variables_dict, status_validos=(0, 1)):
    """
    Conserva los registros del diccionario de variables donde el status_code sea 1 o 2.

    Argumentos:
    - variables_dict (dict): Diccionario devuelto por la función parse_file.
    - status_validos (tuple): Códigos de estado a conservar (por defecto: 1 y 2).

    Retorna:
    - dict: Nuevo diccionario con solo los registros cuyo status_code está en status_validos.
    """
    if not variables_dict:
        raise ValueError("El diccionario de variables está vacío.")

    if filtrar_calidad:
        indices_validos = [i for i, status in enumerate(variables_dict["status_codes"]) if status in status_validos]
        if not indices_validos:
            print("⚠️ No hay registros con status válido.")
        return {key: [values[i] for i in indices_validos] for key, values in variables_dict.items()}
    return variables_dict
    

def convertir_timestamps(array_timestamps):
    """
    Toma un array con timestamps en formato UNIX (como string o int)
    y los devuelve en formato DD.MM.YYYY HH:mm:ss
    """
    fechas_convertidas = []
    for ts in array_timestamps:
        try:
            fechas_convertidas.append(
                datetime.fromtimestamp(int(ts)).strftime("%d.%m.%Y %H:%M:%S")
            )
        except Exception as e:
            print(f"⚠️ Error convirtiendo timestamp {ts}: {e}")
            fechas_convertidas.append("01.01.1970 00:00:00")  # valor por defecto

    return np.array(fechas_convertidas)

#-----------------------------------------------------------------#


#-----------------------PARAMETRIZAR UN UNICO EVENTO ----------------------------------------#
def calcular_parametros_microfisicos(ruta_txt_formateado, vel_terminal, area_efectiva_muestreo, dt_medicion=60):
    """
    Calcula los parametros microfisicos por minuto.

    Args:
    - ruta_txt_formateado: ruta al txt con formato apto PyDSD
    - vel_terminal (array): velocidad terminal que se usara para los calculos de Nd
    - area_efectiva_muestreo (array): area efectiva de muestreo de los bins
    - dt_medicion (int): delta de tiempo entre mediciones del disdrometro en segundos. Por defecto = 60

    Retorna:
    - timestamps: por cada minuto de muestreo
    - Nd (ndarray): Concentracion volumetrica [mm^-1 m^-3]
    - Dm (array): Diámetro medio [mm] por cada minuto de muestreo
    - D0 (array): Diámetro mediano volumétrico [mm] por cada minuto de muestreo
    - Dmax (array): Diámetro máximo [mm] por cada minuto de muestreo
    - Nw (array): Parámetro de interseccion normalizado [mm^-1 m^-3] por cada minuto de muestreo
    - mu (array): Parametro de forma por cada minuto de muestreo
    - LWC (array): Contenido liquido [g mm^-3] por cada minuto de muestreo
    - R (array): Intensidad [mm h^-1] por cada minuto de muestreo
    """
    # Lee txt y crea objeto DSD 
    # FILTRA Raw data usando alguna matriz de filtro (por ahora Hardcoded, ver ParsivelReader.py)
    dsd = pyd.read_parsivel(ruta_txt_formateado)
    
    # Recalcula la DSD para que los parametros sean consistentes con la raw data filtrada
    dsd.calculate_dsd_from_spectrum_columnwise(dt_medicion, vel_terminal, area_efectiva_muestreo) # Recalc Nd
    dsd.calculate_RR() # Recalcula intensidad

    # Calcula parametros D0, Dm, Nw, mu, LWC, Dmax, etc.
    dsd.calculate_dsd_parameterization()
    
    return dsd
#-----------------------------------------------------------------#


#-----------------------CALCULAR PARAMETROS PARA DSD---------------------------#
def calcular_parametros_dsd(ruta_txt_formateado, vel_terminal, area_efectiva_muestreo, duracion_evento, dt_medicion=60):
    """
    Calcula los parametros necesarios para graficar la DSD para un evento dado.

    Args:
    - ruta_txt_formateado: ruta al txt con formato apto PyDSD
    - vel_terminal (array): velocidad terminal que se usara para los calculos de Nd
    - area_efectiva_muestreo (array): area efectiva de muestreo de los bins
    - duracion_evento (int): duracion total del evento en minutos [min]
    - dt_medicion (int): delta de tiempo entre mediciones del disdrometro en segundos. Por defecto = 60

    Retorna:
    - Nd (array): Concentracion volumetrica [mm^-1 m^-3]
    - Dm (float): Diámetro medio [mm]
    - Nw (float): Parámetro de interseccion normalizado [mm^-1 m^-3]
    - mu (float): Parametro de forma
    """
    # Extraer y calcular la matriz total para el evento 
    matriz_evento = extraer_y_sumar_matrices(ruta_txt_formateado)
    
    # Crear un txt virtual y luego contener el diccionario (no genera un txt en archivos)
    dsd_virtual = crear_txt_dsd_virtual(matriz_evento, vel_terminal)
    
    # Recalcula Nd en funcion de la matriz filtrada
    dsd_virtual.calculate_dsd_from_spectrum_columnwise(dt_medicion*duracion_evento, vel_terminal, area_efectiva_muestreo)
    
    # Recalcular intensidad y parametros microfisicos
    dsd_virtual.calculate_RR()
    dsd_virtual.calculate_dsd_parameterization()

    # Guardar parametros de interes
    Nd = ((dsd_virtual.fields["Nd"]["data"]).data) # Tenia un [0] antes
    Dm = (dsd_virtual.fields["Dm"]["data"]).data  # Diámetro medio [mm]
    Nw = (dsd_virtual.fields["Nw"]["data"]).data  # Parámetro de interseccion normalizado [mm^-1 m^-3]
    mu = (dsd_virtual.fields["mu"]["data"]).data  # Parametro de forma
    
    return Nd, Dm, Nw, mu


def extraer_y_sumar_matrices(ruta_txt_formateado):
    """
    Lee un archivo de evento formateado para PyDSD y suma todas las matrices 93 (raw_matrix).
    
    Args:
        ruta_txt_formateado (str): ruta al archivo txt formateado por `formatear_txt_pydsd`.
    
    Returns:
        suma_total (np.ndarray): matriz 32x32 con la suma de todas las raw_matrix del evento.
        matrices_individuales (list): lista con todas las matrices 32x32 como np.arrays.
    """
    matrices_individuales = []

    with open(ruta_txt_formateado, "r", encoding="latin-1") as f:
        for linea in f:
            if linea.startswith("93:"):
                datos_str = linea.replace("93:", "").strip().split(";")
                datos_str = [d.strip() for d in datos_str if d.strip() != ""]  # Eliminar vacíos
                if len(datos_str) != 1024:
                    print(f"❌ Línea con {len(datos_str)} elementos: {datos_str[-5:]}")
                    raise ValueError(f"⚠️ Línea 93 con {len(datos_str)} elementos, se esperaban 1024.")
                matriz = np.array([int(val) for val in datos_str]).reshape((32, 32))
                matrices_individuales.append(matriz)

    if not matrices_individuales:
        raise ValueError("❌ No se encontraron matrices 93 en el archivo.")

    suma_total = np.sum(matrices_individuales, axis=0)
    
    return suma_total


def crear_txt_dsd_virtual(matriz_32x32, vel_caida):
    """
    Crea un archivo .txt temporal con un formato igual al que recibe parsivel_reader.
    Este archivo condensa un evento completo, el objetivo es crear una matriz de 32x32
    global y asi calcular los parametros DSD desde esa matriz para todo el evento. 
    """
    
    # Obtener el contenido como string
    contenido = construir_txt_virtual(matriz_32x32, vel_caida).getvalue()

    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='latin-1', suffix='.txt') as tmp:
        tmp.write(contenido)
        ruta_temporal = tmp.name

    try:
        # Usar el archivo como si fuera uno real
        dsd = pyd.read_parsivel(ruta_temporal)
    finally:
        os.remove(ruta_temporal)  # Eliminar el archivo luego de cargar el objeto

    return dsd

def construir_txt_virtual(matriz_32x32, vel_caida):
    """
    Construye la estructura de un .txt temporal semejante al formato que lee
    parsivel_reader usando los prefijos para identificar las variables.
    """
    ahora = datetime.now()
    fecha = ahora.strftime('%d.%m.%Y')
    hora = ahora.strftime('%H:%M:%S')

    bloque_11 = str(int(matriz_32x32.sum())) + '\n'
    bloque_90 = ';'.join(['-9.999'] * 32) + ';\n'
    bloque_91 = ';'.join(f'{v:.6f}' for v in vel_caida) + ';\n'
    bloque_93 = matriz_a_linea_formateada(matriz_32x32) + '\n'

    contenido = (
        f'21: {fecha}\n'
        f'20: {hora}\n'
        '01: 0.0\n'
        '07: 10.000\n'
        f'11: {bloque_11}'
        f'90: {bloque_90}'
        f'91: {bloque_91}'
        f'93: {bloque_93}'
    )

    return StringIO(contenido)

def matriz_a_linea_formateada(matriz):
    """
    Toma una matriz 32x32 y la formatea como una sola linea flatten donde
    cada 3 caracteres representan un valor de la matriz
    """
    matriz_transpuesta = matriz
    valores = matriz_transpuesta.flatten()
    # Formatear cada número como 'NNN;'
    linea = ''.join(f'{int(v):03d};' for v in valores)
    return linea

#-----------------------------------------------------------------#


#-----------------------EXPORTAR EXCEL CON RESUMEN ESTADISTICO---------------------------#
def exportar_resumen_estadistico(dsd, ruta_txt, timestamp_str):
    """
    Calcula métricas estadísticas para variables de la DSD y exporta a un Excel.

    Args:
        dsd (DropSizeDistribution): Objeto de PyDSD con las variables.
        ruta_txt (str): Ruta al archivo txt formateado.
        timestamp_str (str): Timestamp inicial del evento (para nombrar archivo).

    Returns:
        ruta_salida (str): Ruta del archivo Excel generado.
    """
    # Variables de interés
    variables = ["Dm", "D0", "Dmax", "Nw", "mu", "W", "rain_rate"]

    # Función para limpiar datos (enmascarados o no)
    def get_clean_data(variable):
        if variable in dsd.fields:
            data = dsd.fields[variable]["data"]
            return data.filled(np.nan) if np.ma.is_masked(data) else np.array(data)
        return None

    # Timestamp seguro para nombre
    try:
        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H-%M")
        timestamp_seguro = timestamp_dt.strftime("%Y-%m-%d_%H-%M")
    except ValueError:
        timestamp_seguro = timestamp_str.replace(":", "-").replace(" ", "_")

    # Carpeta de salida (misma que para gráficos)
    carpeta_salida = os.path.join(os.path.dirname(ruta_txt), "resultados")
    os.makedirs(carpeta_salida, exist_ok=True)

    # Diccionario para almacenar resultados
    resultados = {}

    for var in variables:
        valores = get_clean_data(var)
        if valores is None or np.all(np.isnan(valores)):
            continue

        if var == "Nw":
            valores = np.log10(valores)
            nombre_var = "log10Nw"
        else:
            nombre_var = var

        # Calcular métricas
        promedio = np.nanmean(valores)
        moda_val = mode(valores, nan_policy="omit").mode
        moda_val = moda_val[0] if len(moda_val) > 0 else np.nan
        std = np.nanstd(valores)
        minimo = np.nanmin(valores)
        maximo = np.nanmax(valores)
        p5 = np.nanpercentile(valores, 5)
        p95 = np.nanpercentile(valores, 95)

        resultados[nombre_var] = {
            "Promedio": promedio,
            "Moda": moda_val,
            "Desv. Estándar": std,
            "Mínimo": minimo,
            "Máximo": maximo,
            "Percentil 5": p5,
            "Percentil 95": p95
        }

    # Convertir a DataFrame
    df_resultados = pd.DataFrame(resultados).T  # variables como filas

    # Guardar en Excel
    nombre_archivo = f"resumen_estadistico_{timestamp_seguro}.xlsx"
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
    df_resultados.to_excel(ruta_salida, index=True)

    print(f"✅ Resumen estadístico guardado en: {ruta_salida}")
    return ruta_salida

