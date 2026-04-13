import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mne
from tqdm import tqdm
import pickle
from collections import Counter
import seaborn as sns
import scipy.ndimage
from scipy import signal

data_path = '/home/samuel/Documentos/Visual_Reasoning/data/data/'

def procesar_datos_eyelink(file_folder, fname):
    """Carga y procesa el archivo .asc"""
    file_name = os.path.join(file_folder, fname + '.asc')
    # Usamos read_edf
    raw = mne.io.read_raw_eyelink(file_name, apply_offsets=False, verbose=False)
    print(raw.ch_names)
    
    # Extraer eventos
    pd_ann = raw.annotations.to_data_frame(time_format='ms')
    time_fix_cross = pd_ann[pd_ann['description'].str.contains('Fix_Crossstr')]['onset'].array
    time_stim_pres = pd_ann[pd_ann['description'].str.contains('Stim_Presstr')]['onset'].array
    time_keyboard = pd_ann[pd_ann['description'].str.contains('KEYBOARD')]['onset'].array
    keyboard = pd_ann[pd_ann['description'].str.contains('KEYBOARD')]['description'].array
    responses = [int(mensaje.split()[2])-256 for mensaje in keyboard if "KEYBOARD" in mensaje]

    # Extraer datos de posición
    eye_data, time_array = raw.get_data(picks=['xpos_left', 'ypos_left', 'xpos_right', 'ypos_right', 'pupil_left', 'pupil_right'], return_times=True)
    x_left_raw, y_left_raw, x_right_raw, y_right_raw, pupil_left_raw, pupil_right_raw = eye_data
    x_left, y_left, x_right, y_right = x_left_raw.copy(), y_left_raw.copy(), x_right_raw.copy(), y_right_raw.copy()
    pupil_left, pupil_right = pupil_left_raw.copy(), pupil_right_raw.copy()

    # pestañeos (ojo izq)
    mascara_blink = (pupil_left == 0)
    estructura_expansion = np.ones(2 * 50 + 1)
    mascara_dilatada = scipy.ndimage.binary_dilation(mascara_blink, structure=estructura_expansion)
    
    pupil_left[mascara_dilatada] = np.nan
    x_left[mascara_dilatada], y_left[mascara_dilatada] = np.nan, np.nan    
    x_right[mascara_dilatada], y_right[mascara_dilatada] = np.nan, np.nan

    # Cálculo de umbrales sacadas
    x_left_diff, y_left_diff = np.diff(x_left), np.diff(y_left)
    d_left = np.sqrt(x_left_diff**2 + y_left_diff**2)
    umbral_l_per = 90
    umbral_l = np.percentile(d_left[~np.isnan(d_left)], umbral_l_per)

    x_right_diff, y_right_diff = np.diff(x_right), np.diff(y_right)
    d_right = np.sqrt(x_right_diff**2 + y_right_diff**2)
    umbral_r_per = 90
    umbral_r = np.percentile(d_right[~np.isnan(d_right)], umbral_r_per)

    # Identifica cambios de estado
    is_sac_bool = d_left >= umbral_l
    cambios_estado = np.diff(is_sac_bool, prepend=is_sac_bool[0])
    indices_cambio = np.where(cambios_estado)[0]
    indices_cambio = np.append(indices_cambio, len(d_left) - 1)

    eventos = []
    idx_inicio = 0
    estado_actual = is_sac_bool[0] # El estado con el que empezamos

    # Iterar solo sobre los índices donde el estado cambia (muy rápido)
    for idx_fin in indices_cambio:
        if idx_fin == idx_inicio:
            continue
            
        t_inicio = time_array[idx_inicio] * 1000.0
        t_fin = time_array[idx_fin] * 1000.0
        
        if estado_actual: 
            # Es una SACADA (guardamos inicio y fin)
            x_ini, y_ini = x_left[idx_inicio], y_left[idx_inicio]
            x_fin, y_fin = x_left[idx_fin], y_left[idx_fin]
            data_evento = [t_inicio, t_fin, t_fin-t_inicio, x_ini, y_ini, x_fin, y_fin, 1]
            
        else: 
            # Es una FIJACIÓN (guardamos el promedio en ambas posiciones para mantener tu formato)
            segmento_x = x_left[idx_inicio:idx_fin]
            segmento_y = y_left[idx_inicio:idx_fin]
            
            x_mean = np.nanmean(segmento_x) if len(segmento_x) > 0 else np.nan
            y_mean = np.nanmean(segmento_y) if len(segmento_y) > 0 else np.nan
            data_evento = [t_inicio, t_fin, t_fin-t_inicio, x_mean, y_mean, x_mean, y_mean, 0]

        if t_fin-t_inicio > 4:
            eventos.append(data_evento)
            
        idx_inicio = idx_fin
        estado_actual = not estado_actual # Invertimos el estado

    # Convertir a DataFrame de Pandas y exportar a CSV
    columnas = ['time_start_ms', 'time_end_ms', 'long', 'x_start', 'y_start', 'x_end', 'y_end', 'is_saccade']
    df_eventos = pd.DataFrame(eventos, columns=columnas)
    
    # Guardar en CSV sin el índice por defecto de Pandas
    df_eventos.to_csv(os.path.join(file_folder, fname+'_oc_events.csv'), index=False)
    print(df_eventos.head()) # Para que veas las primeras filas en la termina

    # get images
    images_list = get_image_list(file_folder)

    # Create and save data
    datos = {
        "time_array": time_array, 
        "x_left_raw": x_left_raw, "y_left_raw": y_left_raw, "x_right_raw": x_right_raw, "y_right_raw": y_right_raw,
        "x_left": x_left, "y_left": y_left, "x_right": x_right, "y_right": y_right,
        "pupil_left_raw": pupil_left_raw, "pupil_left": pupil_left,
        "images_list": images_list,
        "responses": responses,
        "events": (time_fix_cross, time_stim_pres, time_keyboard)
    }

    newdata_name = os.path.join(file_folder, fname+'.dat')
    print(newdata_name)
    with open(newdata_name, 'wb') as f: # 'wb' = write binary
        pickle.dump(datos, f)

    return datos


def get_image_list(file_folder):
    """Saca los nombres de las imágenes de los archivos MODULO.dat"""

    img_mod_1, img_mod_2, images_list = [], [], []
    for nombre_archivo in os.listdir(file_folder):
        if "MODULO_1" in nombre_archivo:
            with open(os.path.join(file_folder, nombre_archivo), 'r') as f:
                img_mod_1 = [line.strip().strip('"\'') for line in f]
        elif "MODULO_2" in nombre_archivo:
            with open(os.path.join(file_folder, nombre_archivo), 'r') as f:
                img_mod_2 = [line.strip().strip('"\'') for line in f]
    images_list = img_mod_1+img_mod_2

    return images_list


#######################################################################################
################################### Bucle principal ###################################
#######################################################################################
lista_de_carpetas = [nombre for nombre in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, nombre))]
print(f"Las carpetas dentro de '{data_path}' son:")
print(lista_de_carpetas)

for fname in tqdm(lista_de_carpetas, desc="Procesando carpetas"):

    file_folder =  os.path.join(data_path, fname)
    print(fname)
    # Si existe el dato lo carga, sino lo crea
    dat_file = os.path.join(file_folder, fname + '.dat')
    if False:#os.path.exists(dat_file):
        print('Existe')
        with open(dat_file, 'rb') as f:
            datos = pickle.load(f)
    else:
        print('No existe')
        datos = procesar_datos_eyelink(file_folder, fname)


    #print(datos["images_list"])
    #print(datos["responses"])

    
    
print("Proceso completado.")
