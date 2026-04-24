import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. FUNCIONES GRAFICADORAS
# ==========================================

def grafico_comportamiento_temporal(time_raw, x_raw, y_raw, t_fix, t_stim, img_name, save_path, trial):
    """
    Recrea el 'oc_trials'. Genera un gráfico de 2 subplots (X arriba, Y abajo)
    mostrando la posición a lo largo del tiempo, con líneas verticales de eventos.
    """
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 6), layout='constrained', sharex=True)
    fig.suptitle(f'Comportamiento Ocular - Trial: {trial} ({img_name})', fontsize=16)

    # Gráfico X
    ax_x.plot(time_raw, x_raw, color='blue', linewidth=1.5)
    ax_x.set_ylabel('Posición X (píxeles)')
    ax_x.axvline(x=t_fix, color='green', linestyle='--', label='Fix Cross')
    ax_x.axvline(x=t_stim, color='red', linestyle='--', label='Stimulus Onset')
    ax_x.set_facecolor('#d3d3d3') # Fondo gris como en tu código antiguo
    ax_x.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.7)
    ax_x.legend(loc='upper right')

    # Gráfico Y
    ax_y.plot(time_raw, y_raw, color='orange', linewidth=1.5)
    ax_y.set_ylabel('Posición Y (píxeles)')
    ax_y.set_xlabel('Tiempo (segundos)')
    ax_y.axvline(x=t_fix, color='green', linestyle='--')
    ax_y.axvline(x=t_stim, color='red', linestyle='--')
    ax_y.set_facecolor('#d3d3d3')
    ax_y.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.7)

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def grafico_yarbus_trial(x_raw, y_raw, time_raw, img_path, res_pantalla, save_path, trial):
    """
    Recrea el 'yarbus_trial'. Dibuja el scanpath sobre la imagen usando 
    un mapa de colores HSV para representar el avance del tiempo.
    """
    # Cargar la imagen y calcular offset
    try:
        img = Image.open(img_path)
        res_imagen = img.size
        offset_x = (res_pantalla[0] - res_imagen[0]) / 2
        offset_y = (res_pantalla[1] - res_imagen[1]) / 2
    except FileNotFoundError:
        # Si no encuentra la imagen, crea un lienzo gris
        img = Image.new('RGB', (int(res_pantalla[0]), int(res_pantalla[1])), color=(173, 173, 173))
        offset_x, offset_y = 0, 0

    x_corr = x_raw - offset_x
    y_corr = y_raw - offset_y

    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
    ax.imshow(img)
    ax.set_title(f"Yarbus Trial: {trial} (Degradado de tiempo)", fontsize=14)

    # Crear segmentos para colorear la línea según el tiempo
    puntos = np.array([x_corr, y_corr]).T.reshape(-1, 1, 2)
    segmentos = np.concatenate([puntos[:-1], puntos[1:]], axis=1)

    # Normalizar el tiempo de 0 a 1 para el colormap
    tiempo_norm = (time_raw - time_raw.min()) / (time_raw.max() - time_raw.min())
    
    # Crear la colección de líneas con colormap HSV
    lc = LineCollection(segmentos, cmap='hsv', alpha=0.8, linewidths=2)
    lc.set_array(tiempo_norm)
    line = ax.add_collection(lc)

    # Añadir los puntos exactos (fijaciones crudas) superpuestas
    ax.scatter(x_corr, y_corr, c=tiempo_norm, cmap='hsv', s=10, zorder=5)

    ax.axis('off')
    
    # Añadir barra de color
    cbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Tiempo Normalizado (0 = Inicio, 1 = Fin)')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 2. BUCLE PRINCIPAL
# ==========================================

if __name__ == "__main__":
    # Rutas base
    data_path = '/home/samuel/Documentos/Visual_Reasoning/data/processed/'
    img_path_base = '/home/samuel/Documentos/Visual_Reasoning/img_question/img_test/'
    
    # Carpetas de salida para estos gráficos específicos
    out_oc_trials = '/home/samuel/Documentos/Visual_Reasoning/results/oc_trials/'
    out_yarbus = '/home/samuel/Documentos/Visual_Reasoning/results/yarbus_trials/'
    
    os.makedirs(out_oc_trials, exist_ok=True)
    os.makedirs(out_yarbus, exist_ok=True)

    carpetas = [n for n in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, n))]
    
    for fname in tqdm(carpetas, desc="Generando Trazos Oculares"):
        file_folder = os.path.join(data_path, fname)
        dat_file = os.path.join(file_folder, fname + '.dat')
        answ_file = os.path.join(file_folder, fname + '_answers.csv')
        
        if not os.path.exists(dat_file) or not os.path.exists(answ_file):
            continue
            
        with open(dat_file, 'rb') as f:
            datos = pickle.load(f)
            
        df_answ = pd.read_csv(answ_file)
        res_pantalla = datos.get("screen_resolution", (1920, 1080))
        
        # Iterar sobre cada pregunta/trial del sujeto
        for index, row in df_answ.iterrows():
            img_name = row['img_name']
            ruta_imagen = os.path.join(img_path_base, img_name)
            
            # Obtener tiempos en segundos
            t_fix = datos["events"][0][index] / 1000.0  # Aparece cruz
            t_stim = datos["events"][1][index] / 1000.0 # Aparece imagen
            t_fin = datos["events"][2][index] / 1000.0  # Respuesta
            
            # Máscara desde que aparece la cruz de fijación hasta que responde
            mask_completa = (datos["time_array"] >= t_fix) & (datos["time_array"] <= t_fin)
            
            t_raw = datos["time_array"][mask_completa]
            x_raw = datos["x_left"][mask_completa]
            y_raw = datos["y_left"][mask_completa]
            
            # Limpiar NaNs para no romper las gráficas de Yarbus
            mask_nans = ~np.isnan(x_raw) & ~np.isnan(y_raw)
            t_clean = t_raw[mask_nans]
            x_clean = x_raw[mask_nans]
            y_clean = y_raw[mask_nans]
            
            if len(t_clean) < 10:
                continue

            # Nombres de guardado
            nombre_base = f"{fname}_trial_{index:03d}"
            save_oc = os.path.join(out_oc_trials, f"{nombre_base}_temporal.png")
            save_yarbus = os.path.join(out_yarbus, f"{nombre_base}_yarbus.png")
            
            # 1. Graficar Comportamiento Temporal Crudo (oc_trials)
            grafico_comportamiento_temporal(
                t_raw, x_raw, y_raw, t_fix, t_stim, img_name, save_oc, index
            )
            
            # 2. Graficar Yarbus (Degradado temporal sobre estímulo)
            # Para el Yarbus, pasamos el segmento que empieza en t_stim (aislando la búsqueda visual)
            mask_yarbus = (t_clean >= t_stim)
            if np.sum(mask_yarbus) > 10:
                grafico_yarbus_trial(
                    x_clean[mask_yarbus], y_clean[mask_yarbus], t_clean[mask_yarbus], 
                    ruta_imagen, res_pantalla, save_yarbus, index
                )

        # Break temporal para que pruebes con el primer sujeto y valides el diseño
        break