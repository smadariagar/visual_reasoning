import os
import pandas as pd
import numpy as np
import mne
import pickle
import scipy.ndimage
from tqdm import tqdm


def obtener_resolucion_pantalla(asc_file):
    """Busca la resolución en la cabecera del .asc antes de que MNE lo procese."""
    with open(asc_file, 'r') as f:
        for linea in f:
            if 'DISPLAY_COORDS' in linea:
                partes = linea.split()
                # Sumamos 1 porque las coordenadas van de 0 a 1919 (1920 píxeles)
                ancho = int(partes[-2]) + 1  
                alto = int(partes[-1]) + 1   
                return ancho, alto
            # Si llegamos a los datos numéricos, cortamos la búsqueda
            if linea[0].isdigit(): 
                break
    # Valores por defecto en caso de que el archivo no tenga la cabecera
    return 1920, 1080

def get_image_list(file_folder):
    """Saca los nombres de las imágenes de los archivos MODULO.dat"""
    img_mod_1, img_mod_2 = [], []
    for nombre_archivo in os.listdir(file_folder):
        if "MODULO_1" in nombre_archivo:
            with open(os.path.join(file_folder, nombre_archivo), 'r') as f:
                img_mod_1 = [line.strip().strip('"\'') for line in f]
        elif "MODULO_2" in nombre_archivo:
            with open(os.path.join(file_folder, nombre_archivo), 'r') as f:
                img_mod_2 = [line.strip().strip('"\'') for line in f]
    return img_mod_1 + img_mod_2

def procesar_datos_eyelink(file_folder, fname):
    """Carga y procesa el archivo .asc, extrayendo eventos y posiciones oculares."""
    file_name = os.path.join(file_folder, fname + '.asc')

    # Extraer la resolución de la pantalla ANTES de pasarlo a MNE
    ancho_pantalla, alto_pantalla = obtener_resolucion_pantalla(file_name)

    # Procesamiento normal con MNE
    raw = mne.io.read_raw_eyelink(file_name, apply_offsets=False, verbose=False)
    
    # Extraer eventos
    pd_ann = raw.annotations.to_data_frame(time_format='ms')
    time_fix_cross = pd_ann[pd_ann['description'].str.contains('Fix_Crossstr')]['onset'].array
    time_stim_pres = pd_ann[pd_ann['description'].str.contains('Stim_Presstr')]['onset'].array
    time_keyboard = pd_ann[pd_ann['description'].str.contains('KEYBOARD')]['onset'].array
    keyboard = pd_ann[pd_ann['description'].str.contains('KEYBOARD')]['description'].array
    responses = [int(m.split()[2])-256 for m in keyboard if "KEYBOARD" in m]

    # Extraer datos de posición
    eye_data, time_array = raw.get_data(picks=['xpos_left', 'ypos_left', 'xpos_right', 'ypos_right', 'pupil_left', 'pupil_right'], return_times=True)
    x_left, y_left, x_right, y_right = eye_data[0].copy(), eye_data[1].copy(), eye_data[2].copy(), eye_data[3].copy()
    pupil_left = eye_data[4].copy()

    # Procesamiento de pestañeos y dilatación
    mascara_blink = (pupil_left == 0)
    mascara_dilatada = scipy.ndimage.binary_dilation(mascara_blink, structure=np.ones(2 * 50 + 1))
    
    pupil_left[mascara_dilatada] = np.nan
    x_left[mascara_dilatada], y_left[mascara_dilatada] = np.nan, np.nan
    x_right[mascara_dilatada], y_right[mascara_dilatada] = np.nan, np.nan

    # Cálculo dinámico de sacadas
    d_left = np.sqrt(np.diff(x_left)**2 + np.diff(y_left)**2)
    umbral_l = np.percentile(d_left[~np.isnan(d_left)], 90)
    is_sac_bool = d_left >= umbral_l
    cambios_estado = np.diff(is_sac_bool, prepend=is_sac_bool[0])
    indices_cambio = np.append(np.where(cambios_estado)[0], len(d_left) - 1)

    eventos = []
    idx_inicio, estado_actual = 0, is_sac_bool[0]
    for idx_fin in indices_cambio:
        if idx_fin == idx_inicio: continue
        t_inicio, t_fin = time_array[idx_inicio] * 1000.0, time_array[idx_fin] * 1000.0
        
        if estado_actual:
            data_evento = [t_inicio, t_fin, t_fin-t_inicio, x_left[idx_inicio], y_left[idx_inicio], x_left[idx_fin], y_left[idx_fin], 1]
        else:
            x_mean = np.nanmean(x_left[idx_inicio:idx_fin]) if len(x_left[idx_inicio:idx_fin]) > 0 else np.nan
            y_mean = np.nanmean(y_left[idx_inicio:idx_fin]) if len(y_left[idx_inicio:idx_fin]) > 0 else np.nan
            data_evento = [t_inicio, t_fin, t_fin-t_inicio, x_mean, y_mean, x_mean, y_mean, 0]

        if t_fin-t_inicio > 4: eventos.append(data_evento)
        idx_inicio, estado_actual = idx_fin, not estado_actual

    # Exportar datos
    df_eventos = pd.DataFrame(eventos, columns=['time_start_ms', 'time_end_ms', 'long', 'x_start', 'y_start', 'x_end', 'y_end', 'is_saccade'])
    df_eventos.to_csv(os.path.join(file_folder, fname+'_oc_events.csv'), index=False)

    datos = {
        "time_array": time_array, 
        "x_left": x_left, "y_left": y_left, "x_right": x_right, "y_right": y_right,
        "images_list": get_image_list(file_folder),
        "responses": responses,
        "events": (time_fix_cross, time_stim_pres, time_keyboard),
        "screen_resolution": (ancho_pantalla, alto_pantalla)
    }

    with open(os.path.join(file_folder, fname+'.dat'), 'wb') as f:
        pickle.dump(datos, f)
    return datos

if __name__ == "__main__":
    data_path = '/home/samuel/Documentos/Visual_Reasoning/data/processed/'
    carpetas = [n for n in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, n))]
    
    for fname in tqdm(carpetas, desc="Procesando datos .asc"):
        file_folder = os.path.join(data_path, fname)
        dat_file = os.path.join(file_folder, fname + '.dat')
        
        if not os.path.exists(dat_file):
            procesar_datos_eyelink(file_folder, fname)

        break