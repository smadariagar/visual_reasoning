import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

def cargar_datos_preprocesados(file_folder, fname):
    """Carga el .dat y los eventos .csv procesados en el paso 01."""
    with open(os.path.join(file_folder, fname + '.dat'), 'rb') as f:
        datos = pickle.load(f)
    
    eventos_csv = os.path.join(file_folder, fname + '_oc_events.csv')
    df_eventos = pd.read_csv(eventos_csv) if os.path.exists(eventos_csv) else None
    
    return datos, df_eventos

def corregir_coordenadas(x, y, res_pantalla, res_imagen):
    """
    Ajusta las coordenadas del EyeLink al tamaño real de la imagen en pantalla.
    """
    offset_x = (res_pantalla[0] - res_imagen[0]) / 2
    offset_y = (res_pantalla[1] - res_imagen[1]) / 2
    
    x_corr = x - offset_x
    y_corr = y - offset_y
    
    return x_corr, y_corr

def plot_scanpath_and_heatmap(x_ojos, y_ojos, ruta_imagen, res_pantalla, save_path=None):
    """
    Genera una figura con dos subplots: El Scanpath (trayectoria) y el Heatmap (densidad).
    """
    # Cargar la imagen y obtener su resolución real
    img = Image.open(ruta_imagen)
    res_imagen = img.size # (ancho, alto)
    
    # Corregir los datos del ojo (filtrando NaNs por pestañeos)
    mask_validos = ~np.isnan(x_ojos) & ~np.isnan(y_ojos)
    x_val = x_ojos[mask_validos]
    y_val = y_ojos[mask_validos]
    
    x_corr, y_corr = corregir_coordenadas(x_val, y_val, res_pantalla, res_imagen)

    # Preparar la figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Análisis Visual: {os.path.basename(ruta_imagen)}", fontsize=16)

    # --- AX1: SCANPATH ---
    ax1.imshow(img)
    ax1.plot(x_corr, y_corr, color='cyan', alpha=0.5, linewidth=1.5, label='Sacadas') # Líneas
    ax1.scatter(x_corr, y_corr, color='red', alpha=0.7, s=15, label='Fijaciones brutas') # Puntos
    ax1.set_title("Scanpath (Trayectoria Ocular)")
    ax1.axis('off')
    ax1.legend()

    # --- AX2: HEATMAP ---
    ax2.imshow(img)
    # Usamos seaborn para el mapa de densidad superpuesto
    sns.kdeplot(x=x_corr, y=y_corr, ax=ax2, fill=True, cmap="inferno", alpha=0.4, bw_adjust=0.5, thresh=0.05)
    ax2.set_title("Heatmap (Densidad de Atención)")
    ax2.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Rutas (ajústalas a tu entorno)
    processed_data_path = '/home/samuel/Documentos/Visual_Reasoning/data/processed/'
    imagenes_path = '/home/samuel/Documentos/Visual_Reasoning/img_question/img_test' 
    
    carpetas = [n for n in os.listdir(processed_data_path) if os.path.isdir(os.path.join(processed_data_path, n))]
    
    for fname in tqdm(carpetas, desc="Generando visualizaciones"):
        file_folder = os.path.join(processed_data_path, fname)
        
        # Cargar datos base y resolución
        datos, df_eventos = cargar_datos_preprocesados(file_folder, fname)
        res_pantalla = datos.get("screen_resolution", (999, 999))
        res_pantalla = (1920-13.712, 1080-40.466)
        print(res_pantalla)
        
        # Cargar el archivo de respuestas para iterar sobre los trials
        answ_file = os.path.join(file_folder, fname + '_answers.csv')
        if not os.path.exists(answ_file):
            print(f"Falta archivo de respuestas en {fname}")
            continue
            
        df_answ = pd.read_csv(answ_file)
        
        # Iterar por cada trial del sujeto
        for index, row in df_answ.iterrows():
            img_name = row['img_name']
            ruta_imagen = os.path.join(imagenes_path, img_name)
            
            if not os.path.exists(ruta_imagen):
                continue
                
            # --- AJUSTE DE LA VENTANA DE TIEMPO ---
            # datos["events"] = (time_fix_cross, time_stim_pres, time_keyboard)
            # MNE extrajo los eventos en milisegundos, pero time_array está en segundos.
            # Por eso dividimos por 1000.
            
            # Inicio: Aparición del estímulo (+ 0.2s para saltar la latencia de la primera sacada, opcional)
            t_inicio = (datos["events"][1][index] / 1000.0) + 0.2
            
            # Fin: El momento exacto en el que presiona la tecla
            t_fin = datos["events"][2][index] / 1000.0
            
            # Crear la máscara (el filtro de tiempo)
            mask_stim = (datos["time_array"] >= t_inicio) & (datos["time_array"] <= t_fin)
            
            # Cortar los datos espaciales aplicando la máscara
            x_trial = datos["x_left"][mask_stim]
            y_trial = datos["y_left"][mask_stim]
            
            # Si no hay datos suficientes en ese trial, lo saltamos
            if len(x_trial) < 10: 
                continue
            
            # Generar el gráfico con un nombre ordenado
            # Ej: S800_ViRes1a_trial_001_imagen_05.png
            save_file = os.path.join(file_folder, f"{fname}_trial_{index:03d}_{img_name.replace('.png', '')}.png")
            
            plot_scanpath_and_heatmap(
                x_ojos=x_trial, 
                y_ojos=y_trial, 
                ruta_imagen=ruta_imagen, 
                res_pantalla=res_pantalla,
                save_path=save_file
            )
            
        # Dejamos un break temporal para que pruebes con el primer sujeto antes de procesar todos.
        break