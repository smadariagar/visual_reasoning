# idealmente acá por cada sujeto se revisarán los tiempos generados para las fijaciones y la sacadas por trial en función de la distribución de velocidad
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from PIL import Image
import pickle
from tqdm import tqdm

# --- RUTAS ---
img_path_base = '/home/samuel/Documentos/Visual_Reasoning/img_question/img_test/'
data_path = '/home/samuel/Documentos/Visual_Reasoning/data/processed/'

# --- CONSTANTES DE TUS CAJAS ---
CARPETA_PKL = '/home/samuel/Documentos/Visual_Reasoning/CLEVR/img_test_pkl/'
FACTOR_ESCALA = 2
ANCHO_ORIGINAL = 480 
ALTO_ORIGINAL = 320
OFFSET_X = (1920 - (ANCHO_ORIGINAL * FACTOR_ESCALA)) // 2
OFFSET_Y = (1080 - (ALTO_ORIGINAL * FACTOR_ESCALA)) // 2

# ==========================================
# FUNCIONES AUXILIARES PARA MARKOV
# ==========================================
def grafico_trial(x_raw, y_raw, x_left, y_left, time, oc_data, save_path, trial):
     
    estilo_fuente = {'family': 'sans-serif', 'size': 12, 'weight': 'bold'}

    fig, ax = plt.subplots(2, 1, figsize=(13, 5.4), layout='constrained')
    fig.suptitle('Comportamiento Ocular: Posición X e Y, Trial '+trial, fontsize=16, fontweight='bold', fontfamily='sans-serif')

    ax[0].plot(time, x_raw, color='k', alpha=0.5, linestyle='--', linewidth=1)
    ax[0].plot(time, x_left, color='k', linewidth=2)
    
    ax[1].plot(time, y_raw, color='k', alpha=0.5, linestyle='--', linewidth=1)
    ax[1].plot(time, y_left, color='k', linewidth=2)

    num_fijaciones = int(len(oc_data)*1.5)
    print(num_fijaciones)

    colores = cm.hsv(np.linspace(0, 1, num_fijaciones))

    for index, row in oc_data.iterrows():

        if row.iloc[2] >= 10 and row.iloc[7] == 0:
            
            t_start_fix, t_end_fix = row.iloc[0]/1000, row.iloc[1]/1000
            mask_fixation = (time >= t_start_fix) & (time <= t_end_fix)
            
            color_actual = colores[index-oc_data.index[0]]
            ax[0].plot(time[mask_fixation], x_left[mask_fixation], color=color_actual, linewidth=2)
            ax[1].plot(time[mask_fixation], y_left[mask_fixation], color=color_actual, linewidth=2)

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

    ruta_guardado = os.path.join(save_path, trial+'.png')
    fig.savefig(ruta_guardado, dpi=300, bbox_inches='tight', transparent=False)    
    
    plt.close(fig)


def grafico_yarbus_trial(x_raw, y_raw, x_left, y_left, oc_data, img_name, save_path, trial, params=None):
     
    img_original = Image.open(os.path.join(img_path_base, img_name))

    fig, ax = plt.subplots(figsize=(9.6, 5.4), layout='constrained')
    ax.imshow(np.array(img_original))

    ax.plot(x_raw, y_raw, color='k', alpha=0.5, linestyle='--', linewidth=1)
    ax.plot(x_left, y_left, color='k', linewidth=2)

    num_fijaciones = int(len(oc_data)*1.5)
    colores = cm.hsv(np.linspace(0, 1, num_fijaciones))

    for index, row in oc_data.iterrows():
        # Puedes acceder por el índice numérico de la columna o por su nombre
        if row.iloc[2] >= 10 and row.iloc[7] == 0:
            color_actual = colores[index-oc_data.index[0]]

            x_fix, y_fix = row.iloc[3], row.iloc[4]
            if params is not None:
                a, b, m, n, t = params
                x_fix, y_fix = aplicar_transformacion_ocular(x_fix, y_fix, a, b, m, n, t)
                
            ax.scatter(x_fix, y_fix, color=color_actual, linewidths=10)

    ax.axis('off')

    ax.set_xlim(0,1920)
    ax.set_ylim(1080,0)

    ruta_guardado = os.path.join(save_path, trial+'.png')
    fig.savefig(ruta_guardado, dpi=300, bbox_inches='tight', transparent=False)    
    
    plt.close(fig)


def grafico_yarbus_con_cajas(x_raw, y_raw, x_left, y_left, oc_data, img_name, cajas_validas, cajas_id_original, save_path, trial, params=None):
     
    img_original = Image.open(os.path.join(img_path_base, img_name))

    fig, ax = plt.subplots(figsize=(16, 9), layout='constrained')
    ax.imshow(np.array(img_original), extent=[0, 1920, 1080, 0])

    # 1. Dibujar Comportamiento Ocular
    ax.plot(x_raw, y_raw, color='k', alpha=0.5, linestyle='--', linewidth=1)
    ax.plot(x_left, y_left, color='k', linewidth=2)

    num_fijaciones = int(len(oc_data)*1.5)
    colores = cm.hsv(np.linspace(0, 1, num_fijaciones))

    for index, row in oc_data.iterrows():
        if row.iloc[2] >= 10 and row.iloc[7] == 0:
            color_actual = colores[index-oc_data.index[0]]
            
            x_fix = row.iloc[3]
            y_fix = row.iloc[4]
            
            # Aplicar transformación si existen parámetros
            if params is not None:
                a, b, m, n, t = params
                x_fix, y_fix = aplicar_transformacion_ocular(x_fix, y_fix, a, b, m, n, t)
                
            ax.scatter(x_fix, y_fix, color=color_actual, linewidths=10)

    # 2. Dibujar las Cajas (Bounding Boxes)
    for i, bbox in enumerate(cajas_validas):
        x_box, y_box, w, h = bbox
        
        # Crear el rectángulo transparente con borde brillante
        rect = patches.Rectangle(
            (x_box, y_box), w, h, 
            linewidth=2, edgecolor='#00FF00', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Agregar etiqueta de texto (ID original y Estado)
        info_texto = f'ID:{cajas_id_original[i]} | E:{i+1}'
        ax.text(
            x_box, y_box - 5, info_texto, 
            color='black', fontsize=10, weight='bold', 
            bbox=dict(facecolor='#00FF00', alpha=0.7, edgecolor='none', pad=2)
        )

    ax.axis('off')
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)

    ruta_guardado = os.path.join(save_path, trial + '.png')
    fig.savefig(ruta_guardado, dpi=300, bbox_inches='tight', transparent=False)    
    plt.close(fig)


def obtener_cajas_de_imagen(img_name):
    """
    Busca el .pkl asociado a la imagen, filtra los objetos válidos (human_label == 1),
    y devuelve sus coordenadas escaladas al lienzo de 1920x1080.
    """
    cajas_validas = []
    cajas_id_original = []
    
    # 1. Identificar el ID original
    try:
        id_str = img_name.split('_')[0] 
    except Exception:
        return cajas_validas, cajas_id_original

    # 2. Buscar el archivo PKL correspondiente
    archivo_pkl = None
    for pkl in os.listdir(CARPETA_PKL):
        if pkl.endswith(f"{id_str}.pkl"):
            archivo_pkl = pkl
            break
            
    if not archivo_pkl:
        return cajas_validas, cajas_id_original

    # 3. Cargar datos y extraer coordenadas
    ruta_pkl = os.path.join(CARPETA_PKL, archivo_pkl)
    with open(ruta_pkl, 'rb') as f:
        masks_data = pickle.load(f)

    for i, objeto in enumerate(masks_data):
        if objeto.get('human_label', 0) == 1:
            bbox_orig = objeto['bbox']
            
            # Escalar y trasladar
            w_new = bbox_orig[2] * FACTOR_ESCALA
            h_new = bbox_orig[3] * FACTOR_ESCALA
            x_new = (bbox_orig[0] * FACTOR_ESCALA) + OFFSET_X
            y_new = (bbox_orig[1] * FACTOR_ESCALA) + OFFSET_Y
            
            cajas_validas.append([x_new, y_new, w_new, h_new])
            cajas_id_original.append(i) # Guardamos el índice original
            
    return cajas_validas, cajas_id_original


def aplicar_transformacion_ocular(x, y, a, b, m, n, t):
    """Aplica la rotación, escala y offset guardados"""
    theta = np.radians(t)
    cx, cy = 1920 / 2, 1080 / 2 
    
    x_c = x - cx
    y_c = y - cy
    
    x_s = x_c * a
    y_s = y_c * b
    
    x_r = x_s * np.cos(theta) - y_s * np.sin(theta)
    y_r = x_s * np.sin(theta) + y_s * np.cos(theta)
    
    return x_r + cx + m, y_r + cy + n


def obtener_vector_estados(x_array, y_array, cajas_validas):
    """
    Devuelve un vector del mismo tamaño que x_array.
    Contiene 0 si el punto es fondo, o el número de estado (1, 2, 3...) si está en una caja.
    """
    estados = np.zeros(len(x_array), dtype=int)
    
    for i, bbox in enumerate(cajas_validas):
        x_box, y_box, w, h = bbox
        
        en_caja = (x_array >= x_box) & (x_array <= x_box + w) & \
                  (y_array >= y_box) & (y_array <= y_box + h)
        
        estados[en_caja] = i + 1
        
    return estados


# ==========================================
# BUCLE PRINCIPAL
# ==========================================
def recorrer_sujetos():
    
    lista_de_carpetas = [nombre for nombre in os.listdir(data_path) 
                            if os.path.isdir(os.path.join(data_path, nombre))]
    lista_de_carpetas.sort() 
    
    for fname in tqdm(lista_de_carpetas, desc="Sujetos"):
        
        file_folder = os.path.join(data_path, fname)
        dat_file = os.path.join(file_folder, fname + '.dat')
        calib_file = os.path.join(file_folder, fname + '_calib_muestra.dat') 
        answ_file = os.path.join(file_folder, fname + '_answers.csv')
        comp_oc_file = os.path.join(file_folder, fname + '_oc_events.csv')

        if not os.path.exists(dat_file): continue
        print(f"Archivo '{dat_file}' existe")
            
        with open(dat_file, 'rb') as f:
            datos = pickle.load(f)

        df_answ = pd.read_csv(answ_file)
        oc_data = pd.read_csv(comp_oc_file)

        tiene_calibracion = False
        mediana_params = (1.0, 1.0, 0.0, 0.0, 0.0)
        if os.path.exists(calib_file):
            with open(calib_file, 'rb') as f:
                datos_calib = pickle.load(f)
                parametros_dict = datos_calib.get("parametros_muestra", {})
                
                if len(parametros_dict) > 0:
                    # Extraer listas de cada parámetro individual
                    lista_a = [p[0] for p in parametros_dict.values()]
                    lista_b = [p[1] for p in parametros_dict.values()]
                    lista_m = [p[2] for p in parametros_dict.values()]
                    lista_n = [p[3] for p in parametros_dict.values()]
                    lista_t = [p[4] for p in parametros_dict.values()]
                    
                    # Calcular la mediana
                    mediana_params = (
                        np.median(lista_a), np.median(lista_b),
                        np.median(lista_m), np.median(lista_n),
                        np.median(lista_t)
                    )
                    tiene_calibracion = True


        ruta_resultados = os.path.join(file_folder, 'results/')
        ruta_res_trial = os.path.join(ruta_resultados, 'oc_trials/')
        ruta_yarbus_trial = os.path.join(ruta_resultados, 'yarbus_trials/')
        ruta_yarbus_adj = os.path.join(ruta_resultados, 'yarbus_trials_ajustados/')
        ruta_secuencias = os.path.join(ruta_resultados, 'secuencias/')        
        
        for ruta in [ruta_resultados, ruta_res_trial, ruta_yarbus_trial, ruta_yarbus_adj, ruta_secuencias]:
            os.makedirs(ruta, exist_ok=True)

        for index, row in df_answ.iterrows():
            img_name = row['img_name']
            row_idx = index
                  
            try:
                t_ini = datos["events"][1][row_idx]/1000 + 0.1
                t_end = datos["events"][2][row_idx]/1000
                mask_test = (datos["time_array"] >= t_ini) & (datos["time_array"] <= t_end)

                time_trial = datos["time_array"][mask_test]
                
                # Obtener data cruda
                x_raw = datos["x_left_raw"][mask_test]
                y_raw = datos["y_left_raw"][mask_test]

                # Obtener data blink interpol
                x_left = datos["x_left"][mask_test]
                y_left = datos["y_left"][mask_test]

                # Obtener eventos oculares
                col_inicio, col_fin = oc_data.columns[0], oc_data.columns[1]
                mask_oc = (oc_data[col_fin] >= t_ini*1000) & (oc_data[col_inicio] <= t_end*1000)
                oc_data_trial = oc_data[mask_oc].copy()

                cajas_validas, cajas_id_original = obtener_cajas_de_imagen(img_name)

                # grafico_trial(x_raw, y_raw, x_left, y_left, time_trial, oc_data_trial, ruta_res_trial, str(index))
                # grafico_yarbus_trial(x_raw, y_raw, x_left, y_left, oc_data_trial, img_name, ruta_yarbus_trial,  str(index))

                if tiene_calibracion:
                    a_med, b_med, m_med, n_med, t_med = mediana_params
                    
                    # Aplicamos la transformación mediana a todos los ensayos
                    x_raw_adj, y_raw_adj = aplicar_transformacion_ocular(x_raw, y_raw, a_med, b_med, m_med, n_med, t_med)
                    x_left_adj, y_left_adj = aplicar_transformacion_ocular(x_left, y_left, a_med, b_med, m_med, n_med, t_med)

                    # grafico_yarbus_con_cajas(
                    #     x_raw_adj, y_raw_adj, x_left_adj, y_left_adj, 
                    #     oc_data_trial, img_name, 
                    #     cajas_validas, cajas_id_original, 
                    #     ruta_yarbus_adj, str(index) + "_adj", 
                    #     params=mediana_params 
                    # )

                    vector_secuencia = obtener_vector_estados(x_left_adj, y_left_adj, cajas_validas)

                    df_secuencia = pd.DataFrame({
                        'time_s': time_trial,
                        'estado_caja': vector_secuencia
                    })
                    
                    nombre_csv = f"{index}_secuencia.csv"
                    ruta_csv_guardado = os.path.join(ruta_secuencias, nombre_csv)
                    df_secuencia.to_csv(ruta_csv_guardado, index=False)

            except Exception as e:
                print(f"Error extrayendo datos oculares para {img_name}: {e}")
                continue

        break



# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    recorrer_sujetos()