import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from datetime import datetime
from scipy.special import gamma

def graficar_parametros_dsd_con_banda(dsd, ruta_txt, ventana=5):
    """
    Grafica Dm, mu, LWC y rain_rate a lo largo del tiempo, y guarda el gráfico como imagen.

    Args:
        dsd (obj): Objeto DropSizeDistribution de PyDSD.
        ventana (int): Tamaño de la ventana de suavizado para Dm y mu. Default: 5.
        ruta_txt (str): ruta al archivo txt de entrada
        carpeta_salida (str): Carpeta donde se guarda el gráfico. Se crea si no existe.
    """
    # Utilidad para limpiar datos (enmascarados o no)
    def get_clean_data(variable):
        data = dsd.fields[variable]["data"]
        return data.filled(np.nan) if np.ma.is_masked(data) else np.array(data)

    # Extraer variables
    Dm = get_clean_data("Dm")
    mu = get_clean_data("mu")
    LWC = get_clean_data("W")
    RR = get_clean_data("rain_rate")
    
    # Convertir timestamps
    timestamps_raw = dsd.time["data"]
    timestamps = pd.to_datetime(timestamps_raw)

    # Armar DataFrame
    df = pd.DataFrame({
        "Dm": Dm,
        "mu": mu,
        "LWC": LWC,
        "RR": RR
    }, index=timestamps)

    # Media móvil para bandas
    df_smooth = df[["Dm", "mu"]].rolling(window=ventana, center=True, min_periods=1).mean()

    # 🗂️ Crear carpeta donde se guardara el gráfico
    carpeta_salida = os.path.join(os.path.dirname(ruta_txt), "graficos")
    os.makedirs(carpeta_salida, exist_ok=True)

    # Generar nombre del archivo con primer timestamp
    fecha_inicio = timestamps[0].strftime("%Y-%m-%d %H-%M")
    nombre_archivo = f"parametros_DSD_{fecha_inicio}.png"
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)

    # Graficar
    fig, axs = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

    # --- Dm ---
    axs[0].plot(df.index, df["Dm"], label="Dm", color="tab:blue", alpha=0.4)
    axs[0].plot(df.index, df_smooth["Dm"], label=f"Media móvil ({ventana})", color="tab:blue")
    # Parche fantasma para la leyenda
    banda_leyenda = mpatches.Patch(color='tab:blue', alpha=0.2, label='Intervalo media movil (±5%)')
    # Añadimos el fill_between sin etiqueta
    axs[0].fill_between(df.index, df_smooth["Dm"]*0.95, df_smooth["Dm"]*1.05, color="tab:blue", alpha=0.2)
    # Obtenemos las etiquetas y manejadores actuales
    handles, labels = axs[0].get_legend_handles_labels()
    # Añadimos el parche fantasma a la lista de manejadores y su etiqueta a la lista de etiquetas
    handles.append(banda_leyenda)
    labels.append(banda_leyenda.get_label())
    # Creamos la leyenda con los manejadores y etiquetas actualizados
    axs[0].set_title("Diámetro medio volumétrico (Dm)")
    axs[0].legend(handles=handles, labels=labels, loc="upper left")
    axs[0].grid(True)
    
    # --- mu ---
    axs[1].plot(df.index, df["mu"], label="μ", color="tab:orange", alpha=0.4)
    axs[1].plot(df.index, df_smooth["mu"], label=f"Media móvil ({ventana})", color="tab:orange")
    # Creamos un parche "fantasma" para la leyenda de la banda
    banda_leyenda_mu = mpatches.Patch(color='tab:orange', alpha=0.2, label='Intervalo media movil (±5%)')
    # Dibujamos la banda sin etiqueta para evitar duplicados en la leyenda
    axs[1].fill_between(df.index, df_smooth["mu"]*0.95, df_smooth["mu"]*1.05, color="tab:orange", alpha=0.2)
    # Obtenemos las etiquetas y manejadores actuales
    handles, labels = axs[1].get_legend_handles_labels()
    # Añadimos el parche a la lista de manejadores y su etiqueta a la lista de etiquetas
    handles.append(banda_leyenda_mu)
    labels.append(banda_leyenda_mu.get_label())
    # Creamos la leyenda con los manejadores y etiquetas actualizados
    axs[1].legend(handles=handles, labels=labels, loc="upper left")  
    axs[1].set_ylabel("μ")
    axs[1].set_title("Parámetro de forma μ")
    axs[1].grid(True)


    # --- LWC ---
    axs[2].plot(df.index, df["LWC"], label="LWC", color="tab:green")
    axs[2].set_ylabel("LWC [g/m³]")
    axs[2].set_title("Contenido de agua líquida (LWC)")
    axs[2].legend(loc="upper left")
    axs[2].grid(True)

    # --- Rain Rate ---
    axs[3].plot(df.index, df["RR"], label="Rain Rate", color="tab:purple")
    axs[3].set_ylabel("R [mm/h]")
    axs[3].set_title("Intensidad de precipitación (Rain Rate)")
    axs[3].set_xlabel("Tiempo")
    axs[3].legend(loc="upper left")
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()

    print(f"✅ Gráfico de parametros guardado en: {ruta_salida}")

    

def graficar_DSD(Nw, Dm, mu, D_bins, Nd_bins, ruta_txt, timestamp_str, D_min=0.062):
    """
    Grafica la DSD y la distribución gamma ajustada, y guarda el gráfico en la carpeta 'graficos'
    donde se encuentra el txt del evento.

    Args:
        Nw (float): parámetro Nw
        Dm (float): diámetro medio [mm]
        mu (float): parámetro de forma mu
        D_bins (array): array de diámetros por bin [mm]
        Nd_bins (array): array de N(D) por bin
        ruta_txt (str): ruta al archivo txt de entrada
        timestamp_str (str): timestamp con formato '%Y-%m-%d %H:%M'
        D_min (float): diámetro mínimo para graficar
    """

    def calcular_f_mu(mu):
        return (6 / (4**4)) * ((4 + mu)**(mu + 4)) / gamma(mu + 4)

    def N_D(D, Nw, Dm, mu):
        f_mu = calcular_f_mu(mu)
        return Nw * f_mu * (D / Dm)**mu * np.exp(-(4 + mu) * D / Dm)

    # Crear curva continua
    ultimo_bin = np.max(np.nonzero(Nd_bins))
    D_max = D_bins[ultimo_bin] + (D_bins[1] - D_bins[0]) / 2
    D = np.linspace(D_min, D_max)
    N_vals = N_D(D, Nw, Dm, mu)

    # Graficar
    plt.figure(figsize=(9, 5))
    label_gamma = (
        "Distribución gamma ajustada\n"
        f"$N_w = {float(Nw):.1e}\\ \\mathrm{{mm}}^{{-1}}\\,\\mathrm{{m}}^{{-3}}$\n"
        f"$D_m = {float(Dm):.2f}\\ \\mathrm{{mm}}$\n"
        f"$\\mu = {float(mu):.2f}$"
    )
    plt.plot(D, N_vals, label=label_gamma, color='blue')

    # Añadir barras de Nd
    ancho_bin = D_bins[1] - D_bins[0]
    plt.bar(D_bins, Nd_bins, width=ancho_bin*0.8, alpha=0.5, label='Observados', color='orange', edgecolor='k')

    # Etiquetas
    plt.xlabel('Diámetro [mm]', fontsize=13)
    plt.ylabel('$DSD$ [$\\mathrm{mm}^{-1}\\,\\mathrm{m}^{-3}$]', fontsize=13)
    plt.xticks(D_bins, labels=[f"{d:.3f}" for d in D_bins], rotation=90)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim(D_min, D_max)

    # 🗂️ Guardar gráfico
    carpeta_salida = os.path.join(os.path.dirname(ruta_txt), "graficos")
    os.makedirs(carpeta_salida, exist_ok=True)

    timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H-%M")
    nombre_grafico = f"DSD_{timestamp_dt.strftime('%Y-%m-%d %H-%M')}.png"
    ruta_salida = os.path.join(carpeta_salida, nombre_grafico)

    plt.savefig(ruta_salida, dpi=300)
    plt.close()

    print(f"✅ Gráfico de ajuste DSD guardado en: {ruta_salida}")


def graficar_violines_individuales_dsd(dsd, ruta_txt, timestamp_str):
    """
    Genera un único gráfico con violines para cada variable del objeto DropSizeDistribution.
    Los gráficos se colocan en una grilla de 2x3 y se guardan en la carpeta "graficos".

    Args:
        dsd (DropSizeDistribution): Objeto de PyDSD con las variables a graficar.
        ruta_txt (str): Ruta al archivo txt de entrada.
        timestamp_str (str): Timestamp inicial del evento (usado en el nombre del archivo).

    Returns:
        None
    """
    etiquetas_bonitas = {
        "mu": r"$\mu$",
        "Dm": r"$D_{\mathrm{m}}$ [mm]",
        "Dmax": r"$D_{\mathrm{max}}$ [mm]",
        "Nw": r"$N_w$ [$\mathrm{mm}^{-1}\,\mathrm{m}^{-3}$]",
        "W": r"$Contenido Líquido$ [g$\cdot$m$^{-3}$]",
        "rain_rate": r"$Intensidad$ [mm$\cdot$h$^{-1}$]"
    }

    # 🗂️ Carpeta de salida
    carpeta_salida = os.path.join(os.path.dirname(ruta_txt), "graficos")
    os.makedirs(carpeta_salida, exist_ok=True)

    # Timestamp seguro para nombres de archivo
    try:
        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H-%M")
        timestamp_seguro = timestamp_dt.strftime("%Y-%m-%d_%H-%M")
    except ValueError:
        timestamp_seguro = timestamp_str.replace(":", "-").replace(" ", "_")

    # Función auxiliar
    def get_clean_data(variable):
        if variable in dsd.fields:
            data = dsd.fields[variable]["data"]
            return data.filled(np.nan) if np.ma.is_masked(data) else np.array(data)
        return None

    # Crear figura general con subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, variable in enumerate(etiquetas_bonitas.keys()):
        valores = get_clean_data(variable)
        if valores is None:
            continue

        df = pd.DataFrame({
            "valor": valores,
            "variable": [variable] * len(valores)
        })

        etiqueta = etiquetas_bonitas[variable]

        sns.violinplot(
            data=df, x="variable", y="valor",
            inner="quart", color="skyblue", cut=0, ax=axes[i]
        )
        axes[i].set_xlabel("")
        axes[i].set_ylabel(etiqueta, fontsize=12)
        axes[i].set_xticks([])

        # Escala logarítmica para estas variables
        if variable in ["W", "rain_rate", "Nw"]:
            axes[i].set_yscale("log")

        axes[i].grid(True, linestyle="--", alpha=0.3)

    # Ajustar diseño y guardar
    plt.tight_layout()
    nombre_archivo = f"grafico_violin_{timestamp_seguro}.png"
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print(f"✅ Gráfico guardado en: {ruta_salida}")