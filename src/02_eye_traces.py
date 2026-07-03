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

def grafico_comportamiento_temporal(x_raw, y_raw, x_left, y_left, time, t_stim, oc_data, save_path, trial):
    """
    Recrea el 'oc_trials'. Genera un gráfico de 2 subplots (X arriba, Y abajo)
    mostrando la posición a lo largo del tiempo, con líneas verticales de eventos.
    """
    estilo_fuente = {'family': 'sans-serif', 'size': 12, 'weight': 'bold'}

    fig, ax = plt.subplots(2, 1, figsize=(13, 5.4), layout='constrained')
    fig.suptitle('Comportamiento Ocular: Posición X e Y, Trial '+str(trial), fontsize=16, fontweight='bold', fontfamily='sans-serif')

    ax[0].plot(time, x_raw, color='k', alpha=0.5, linestyle='--', linewidth=1)
    ax[0].plot(time, x_left, color='k', linewidth=2)
    
    ax[1].plot(time, y_raw, color='k', alpha=0.5, linestyle='--', linewidth=1)
    ax[1].plot(time, y_left, color='k', linewidth=2)

    num_fijaciones = int(len(oc_data)*1.5)
    #print(num_fijaciones)

    colores = cm.hsv(np.linspace(0, 1, num_fijaciones))

    for index, row in oc_data.iterrows():
        if row.iloc[2] >= 10 and row.iloc[7] == 0:
            t_start_fix, t_end_fix = row.iloc[0]/1000, row.iloc[1]/1000
            mask_fixation = (time >= t_start_fix) & (time <= t_end_fix)
            
            color_actual = colores[index-oc_data.index[0]]
            ax[0].plot(time[mask_fixation], x_left[mask_fixation], color=color_actual, linewidth=2)
            ax[1].plot(time[mask_fixation], y_left[mask_fixation], color=color_actual, linewidth=2)

    for axis in ax:
        y_min, y_max = axis.get_ylim()
        axis.vlines(x=t_stim, ymin=y_min, ymax=y_max, color='k', 
                    linestyle='--', linewidth=2, label='Stimulus Onset', zorder=10)
        center_val = 960 if axis == ax[0] else 540
        axis.axhline(y=center_val, color='g', linestyle=':', alpha=0.5)
    ax[0].set_ylabel('Posición X (px)', fontdict=estilo_fuente)
    ax[1].set_ylabel('Posición Y (px)', fontdict=estilo_fuente)
    ax[1].set_xlabel('Tiempo (s)', fontdict=estilo_fuente)
    
    grosor_linea = 2.0
    for eje in ax:
        eje.tick_params(axis='both', which='major', labelsize=10, width=grosor_linea)
        
        for borde in eje.spines.values():
            borde.set_linewidth(grosor_linea)
        eje.spines['top'].set_visible(False)
        eje.spines['right'].set_visible(False)

    fig.savefig(save_path, dpi=400, bbox_inches='tight', transparent=False)    
    
    plt.close(fig)



# ==========================================
# 2. BUCLE PRINCIPAL
# ==========================================

if __name__ == "__main__":
    
    # Rutas base
    data_path = '/home/samuel/Documentos/Visual_Reasoning/data/processed/'
    img_path_base = '/home/samuel/Documentos/Visual_Reasoning/img_question/img_test/'
    
    carpetas = [n for n in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, n))]
    
    for fname in tqdm(carpetas, desc="Generando Trazos Oculares"):
        file_folder = os.path.join(data_path, fname)
        dat_file = os.path.join(file_folder, fname + '.dat')
        answ_file = os.path.join(file_folder, fname + '_answers.csv')
        comp_oc_file = os.path.join(file_folder, fname + '_oc_events.csv')

        ruta_resultados = os.path.join(file_folder, 'results/')
        ruta_res_trial = os.path.join(ruta_resultados, 'oc_trials/')     
        
        for ruta in [ruta_resultados, ruta_res_trial]:
            os.makedirs(ruta, exist_ok=True)    

        if not os.path.exists(dat_file) or not os.path.exists(answ_file):
            continue
            
        with open(dat_file, 'rb') as f:
            datos = pickle.load(f)
            
        df_answ = pd.read_csv(answ_file)
        oc_data = pd.read_csv(comp_oc_file)

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
            x_l_raw = datos["x_left_raw"][mask_completa]
            y_l_raw = datos["y_left_raw"][mask_completa]
            x_l = datos["x_left"][mask_completa]
            y_l = datos["y_left"][mask_completa]
            
            # Limpiar NaNs para no romper las gráficas de Yarbus
            mask_nans = ~np.isnan(x_l) & ~np.isnan(y_l)
            t_clean = t_raw[mask_nans]
            x_l_clean = x_l_raw[mask_nans]
            y_l_clean = y_l_raw[mask_nans]
            
            # Obtener eventos oculares
            col_inicio, col_fin = oc_data.columns[0], oc_data.columns[1]
            mask_oc = (oc_data[col_fin] >= t_fix*1000) & (oc_data[col_inicio] <= t_fin*1000)
            oc_data_trial = oc_data[mask_oc].copy()

            if len(t_clean) < 10:
                continue

            # Nombres de guardado
            nombre_base = f"{fname}_trial_{index:03d}"
            save_oc = os.path.join(ruta_res_trial, f"{nombre_base}_temporal.png")
            
            # Graficar Comportamiento Temporal Crudo (oc_trials)
            grafico_comportamiento_temporal(x_l_raw, y_l_raw, x_l, y_l, t_raw, t_stim, oc_data_trial, save_oc, index)
            
        #break