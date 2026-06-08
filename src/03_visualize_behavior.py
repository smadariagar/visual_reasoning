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

def grafico_yarbus_trial(x_raw, y_raw, oc_data, img_path, screen_res, save_path, trial):
    """
    Plots the full scanpath as a solid black line and overlays colored circles
    representing individual fixations extracted from the events dataframe.
    Circle sizes are proportional to fixation duration.
    """
    try:
        # 1. Background image setup
        img = Image.open(img_path)
        img_res = img.size

        # Calculate offset to center the image on the screen coordinates
        offset_x = (screen_res[0] - img_res[0]) / 2
        offset_y = (screen_res[1] - img_res[1]) / 2

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img, extent=[0, img_res[0], img_res[1], 0])

        # 2. Adjust raw trajectory coordinates for the black line
        x_adj = x_raw - offset_x
        y_adj = y_raw - offset_y

        # 3. Draw the full trajectory (Solid Black Line)
        ax.plot(x_adj, y_adj, color='black', linewidth=1.2, alpha=0.5, zorder=1)

        # 4. Draw Fixations (Colored Circles based on oc_data)
        # Filter oc_data: column 7 (is_saccade) == 0 and column 2 (duration) >= 10ms
        is_fixation = (oc_data.iloc[:, 7] == 0) & (oc_data.iloc[:, 2] >= 10)
        fixations_df = oc_data[is_fixation]

        if not fixations_df.empty:
            # Extract centroid coordinates (columns 3 and 4 are x_mean and y_mean)
            fix_x = fixations_df.iloc[:, 3] - offset_x
            fix_y = fixations_df.iloc[:, 4] - offset_y

            # Extract start times for the color gradient
            fix_times = fixations_df.iloc[:, 0]

            # Normalize time between 0.0 and 1.0 for the colormap
            time_range = fix_times.max() - fix_times.min()
            t_norm = (fix_times - fix_times.min()) / (time_range if time_range > 0 else 1)

            # Calculate circle sizes based on fixation duration (column 2)
            # Base size of 30, expanding up to +150 based on relative duration
            durations = fixations_df.iloc[:, 2]
            circle_sizes = 30 + (durations / durations.max()) * 150 if durations.max() > 0 else 80

            # Scatter plot for fixations
            sc = ax.scatter(fix_x, fix_y, c=t_norm, cmap='hsv', s=circle_sizes,
                            edgecolors='black', linewidths=1.5, alpha=0.9, zorder=2)

            # Colorbar configuration (fraction and pad keep it proportional to the image)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
            cbar.set_label('Normalized Time (0 = Start, 1 = End)', rotation=270, labelpad=15)

        # 5. Aesthetics and saving
        ax.set_title(f'Yarbus Trial: {trial} (Scanpath & Fixations)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, img_res[0])
        ax.set_ylim(img_res[1], 0) # Inverted Y-axis so 0,0 is top-left
        ax.axis('off')

        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

    except Exception as e:
        print(f"Error processing trial {trial}: {e}")
        plt.close()

def grafico_heat_map(x_gaze, y_gaze, img_path, screen_res, save_path=None):
    """
    Generates a single figure displaying the Heatmap (density of visual attention) 
    over the original stimulus image using Seaborn's KDE plot.
    """
    # Load the image and get its real resolution
    img = Image.open(img_path)
    img_res = img.size # (width, height)
    
    # Filter valid eye data (removing NaNs caused by blinks)
    valid_mask = ~np.isnan(x_gaze) & ~np.isnan(y_gaze)
    x_val = x_gaze[valid_mask]
    y_val = y_gaze[valid_mask]
    
    # Correct coordinates based on screen and image offset
    # Note: Make sure your 'corregir_coordenadas' function is available or translated
    x_corr, y_corr = corregir_coordenadas(x_val, y_val, screen_res, img_res)

    # Setup the figure for a single plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Display the background image
    ax.imshow(img)
    
    # Generate the KDE heatmap overlay
    sns.kdeplot(
        x=x_corr, 
        y=y_corr, 
        ax=ax, 
        fill=True, 
        cmap="inferno", 
        alpha=0.4, 
        bw_adjust=0.5, 
        thresh=0.05
    )
    
    # Aesthetics and titles
    ax.set_title(f"Heatmap: {os.path.basename(img_path)}", fontsize=16, fontweight='bold')
    ax.axis('off') # Hide axes for a cleaner look

    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


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
        ruta_yarbus_trial = os.path.join(ruta_resultados, 'yarbus_trials/')
        ruta_yarbus_adj = os.path.join(ruta_resultados, 'yarbus_trials_ajustados/')
        ruta_heat_map = os.path.join(ruta_resultados, 'heat_maps/') 
                
        for ruta in [ruta_resultados, ruta_res_trial, ruta_yarbus_trial, ruta_yarbus_adj, ruta_heat_map]:
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
            save_yarbus = os.path.join(ruta_yarbus_trial, f"{nombre_base}_yarbus.png")
            save_heatmap = os.path.join(ruta_heat_map, f"{nombre_base}_heatmap.png")
             
            # 1. Graficar Yarbus (Degradado temporal sobre estímulo)
            # Para el Yarbus, pasamos el segmento que empieza en t_stim (aislando la búsqueda visual)
            mask_yarbus = (t_clean >= t_stim)
            mask_oc = (oc_data[col_fin] >= t_stim*1000) & (oc_data[col_inicio] <= t_fin*1000)
            oc_data_trial = oc_data[mask_oc].copy()

            # if np.sum(mask_yarbus) > 10:
            #     grafico_yarbus_trial(x_l_clean[mask_yarbus], y_l_clean[mask_yarbus],
            #         oc_data_trial, ruta_imagen, res_pantalla, save_yarbus, index)
                

            # 2.
            grafico_heat_map(x_l_clean[mask_yarbus], y_l_clean[mask_yarbus],
                    ruta_imagen, res_pantalla, save_heatmap)


        # Break temporal para que pruebes con el primer sujeto y valides el diseño
        #break