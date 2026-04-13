# Markov Chains
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
from PIL import Image
import pickle
from tqdm import tqdm

# --- RUTAS ---
img_path_base = '/home/samuel/Documentos/Visual_Reasoning/img_question/img_test/'
data_path = '/home/samuel/Documentos/Visual_Reasoning/data/data/'
pkl_bbox_path = '/home/samuel/Documentos/Visual_Reasoning/CLEVR/img_test_pkl/'

SCALE_FACTOR = 2
X_OFFSET = 480
Y_OFFSET = 220
# Definimos el tamaño del lienzo de pantalla utilizado
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# ==========================================
# FUNCIONES AUXILIARES PARA MARKOV
# ==========================================
def punto_en_caja(x, y, bbox):
    """Verifica si un punto (x,y) está dentro de un bbox (x_min, y_min, w, h)"""
    bx, by, bw, bh = bbox
    return (bx <= x <= bx + bw) and (by <= y <= by + bh)

def calcular_matriz_transicion(secuencia, num_estados):
    """Calcula la matriz de probabilidades de transición de Markov"""
    # Crear matriz de conteos llena de ceros
    matriz_conteos = np.zeros((num_estados, num_estados))
    
    # Contar las transiciones de S(t) a S(t+1)
    for i in range(len(secuencia) - 1):
        estado_actual = int(secuencia[i])
        estado_siguiente = int(secuencia[i+1])
        matriz_conteos[estado_actual, estado_siguiente] += 1
        
    # Normalizar para obtener probabilidades (evitando división por cero)
    sumas_filas = matriz_conteos.sum(axis=1, keepdims=True)
    # Reemplazamos ceros por unos en la suma para no dividir por cero (esa fila quedará en 0)
    sumas_filas[sumas_filas == 0] = 1 
    
    matriz_probabilidades = matriz_conteos / sumas_filas
    return matriz_probabilidades

def ajustar_bbox(bbox_original, scale, x_off, y_off):
    """
    Ajusta un bbox original (x, y, w, h) según la transformación de pantalla.
    """
    x_min, y_min, w, h = bbox_original
    adj_x = (x_min * scale) + x_off
    adj_y = (y_min * scale) + y_off
    adj_w = w * scale
    adj_h = h * scale
    return (adj_x, adj_y, adj_w, adj_h)

def aplicar_transformacion_ocular(x, y, a, b, m, n, t):
    """Aplica la rotación, escala y offset guardados en revision_v2"""
    theta = np.radians(t)
    cx, cy = 1920 / 2, 1080 / 2 
    
    x_c = x - cx
    y_c = y - cy
    
    x_s = x_c * a
    y_s = y_c * b
    
    x_r = x_s * np.cos(theta) - y_s * np.sin(theta)
    y_r = x_s * np.sin(theta) + y_s * np.cos(theta)
    
    return x_r + cx + m, y_r + cy + n

# ==========================================
# BUCLE PRINCIPAL MODIFICADO
# ==========================================
def cadenas_markov_sujetos():
    
    lista_de_carpetas = [nombre for nombre in os.listdir(data_path) 
                            if os.path.isdir(os.path.join(data_path, nombre))]
    lista_de_carpetas.sort() 
    
    # Diccionario para guardar todas las matrices resultantes por sujeto e imagen
    resultados_markov = {}
    
    for fname in tqdm(lista_de_carpetas, desc="Sujetos"):
        
        file_folder = os.path.join(data_path, fname)
        dat_file = os.path.join(file_folder, fname + '.dat')
        answ_file = os.path.join(file_folder, fname + '_answers.csv')

        if not os.path.exists(dat_file): continue
            
        with open(dat_file, 'rb') as f:
            datos = pickle.load(f)

        df_answ = pd.read_csv(answ_file)
        resultados_markov[fname] = {}

        for index, row in df_answ.iterrows():
            img_name = row['img_name']
            row_idx = index
            
            # 1. Cargar las cajas para esta imagen
            nombre_pkl_cajas = f"CLEVR_test_{img_name.split('_')[0].zfill(6)}.pkl"
            ruta_pkl_cajas = os.path.join(pkl_bbox_path, nombre_pkl_cajas)
            
            if not os.path.exists(ruta_pkl_cajas):
                continue # Si no hay archivo de cajas para esta imagen, saltar
                        
            with open(ruta_pkl_cajas, 'rb') as f:
                cajas_data = pickle.load(f)
                
            cajas_validas = []
            cajas_id_original = [] # Guardamos el ID original para la visualización
            
            for i, obj in enumerate(cajas_data):
                # Verificamos si es válida (human_label == 1), default 0 si no existe
                if obj.get('human_label', 0) == 1:
                    bbox_orig = obj['bbox']
                    # Aplicamos la transformación espacial
                    bbox_adj = ajustar_bbox(bbox_orig, SCALE_FACTOR, X_OFFSET, Y_OFFSET)
                    cajas_validas.append(bbox_adj)
                    cajas_id_original.append(i)
                    
            num_estados = len(cajas_validas) + 1 # +1 por el "Estado 0" (Fondo)
            
            # --- 2. EXTRAER COMPORTAMIENTO OCULAR CORREGIDO ---
            try:
                t_ini = datos["events"][1][row_idx]/1000 + 0.2
                t_end = datos["events"][2][row_idx]/1000
                mask_test = (datos["time_array"] >= t_ini) & (datos["time_array"] <= t_end)
                
                # Obtener data cruda
                x_raw = datos["x_left"][mask_test]
                y_raw = datos["y_left"][mask_test]
                
                try:
                    x_fix_raw = datos["x_left_0"][mask_test[1:]]
                    y_fix_raw = datos["y_left_0"][mask_test[1:]]
                except:
                    x_fix_raw = datos["x_left_0"][mask_test]
                    y_fix_raw = datos["y_left_0"][mask_test]

                # ¡FORZAR CORRECCIÓN V2!
                revisiones = datos.get("revision_v2", {})
                if row_idx in revisiones:
                    a, b, m, n, t = revisiones[row_idx]
                    x_adj, y_adj = aplicar_transformacion_ocular(x_raw, y_raw, a, b, m, n, t)
                    x_fix_adj, y_fix_adj = aplicar_transformacion_ocular(x_fix_raw, y_fix_raw, a, b, m, n, t)
                else:
                    # Si por algún motivo no hay revisión guardada, usamos los crudos o los pre-ajustados
                    x_adj = datos.get("x_left_adjusted", x_raw)[mask_test]
                    y_adj = datos.get("y_left_adjusted", y_raw)[mask_test]
                    
                    try:
                        x_fix_adj = datos.get("x_left_0_adjusted", x_fix_raw)[mask_test[1:]]
                        y_fix_adj = datos.get("y_left_0_adjusted", y_fix_raw)[mask_test[1:]]
                    except:
                        x_fix_adj = datos.get("x_left_0_adjusted", x_fix_raw)[mask_test]
                        y_fix_adj = datos.get("y_left_0_adjusted", y_fix_raw)[mask_test]

            except Exception as e:
                print(f"Error extrayendo datos oculares para {img_name}: {e}")
                continue

            # --- NUEVO/CORREGIDO: VISUALIZAR CAJAS AJUSTADAS SOBRE LIENZO USADO ---
            ruta_imagen = os.path.join(img_path_base, img_name)
            
            if os.path.exists(ruta_imagen):
                img = Image.open(ruta_imagen)
                
                # Replicamos la configuración del lienzo del Código 2 (1920x1080)
                # figsize=(16, 9) da una relación de aspecto 16:9
                fig, ax = plt.subplots(figsize=(16, 9))
                
                # Mostramos la imagen ajustando su extensión al lienzo de 1920x1080
                ax.imshow(img, extent=[0, SCREEN_WIDTH, SCREEN_HEIGHT, 0])
                
                # Dibujar cada caja válida AJUSTADA
                # Usamos la estética del Código 1 (verde brillante) para legibilidad
                for i, bbox in enumerate(cajas_validas):
                    x, y, w, h = bbox
                    # Crear el rectángulo
                    rect = patches.Rectangle(
                        (x, y), w, h, 
                        linewidth=2, 
                        edgecolor='#00FF00', # Verde brillante
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Agregar texto: ID original de caja y número de Estado Markov
                    # i+1 porque el 0 es el fondo
                    info_texto = f'ID:{cajas_id_original[i]}\nE:{i+1}'
                    ax.text(
                        x, y - 5, 
                        info_texto, 
                        color='black', fontsize=8, weight='bold', 
                        bbox=dict(facecolor='#00FF00', alpha=0.7, edgecolor='none', pad=1)
                    )

                # Comportamiento ocular    
                ax.plot(x_adj, y_adj, 'r-', linewidth=1, alpha=0.5, label='Mirada')
                ax.plot(x_fix_adj, y_fix_adj, 'k-', linewidth=1.5, alpha=0.8, label='Fijaciones')
                
                if len(x_adj) > 0: 
                    # Asegurarnos de tomar el primer valor de manera segura
                    x0 = x_adj.iloc[0] if hasattr(x_adj, 'iloc') else x_adj[0]
                    y0 = y_adj.iloc[0] if hasattr(y_adj, 'iloc') else y_adj[0]
                    ax.plot(x0, y0, 'bo', markersize=5, label='Inicio')
                
                ax.set_title(f"Sujeto: {fname} | Img: {img_name} | {len(cajas_validas)} obj", fontsize=14)
                ax.set_xlim(0, SCREEN_WIDTH)
                ax.set_ylim(SCREEN_HEIGHT, 0)
                ax.axis('off')
                ax.legend(loc='upper right')
                
                plt.show() 
                plt.close(fig)
 
            else:
                print(f"Advertencia: No se encontró la imagen {ruta_imagen}")
            # ----------------------------------------------------------------------
            
            # 2. Extraer datos oculares del trial
            try:
                t_ini = datos["events"][1][row_idx]/1000 + 0.2
                t_end = datos["events"][2][row_idx]/1000
                mask_test = (datos["time_array"] >= t_ini) & (datos["time_array"] <= t_end)
                
                # Usar las fijaciones ajustadas (más precisas para Markov)
                # Si tienes x_left_0_adjusted, úsalo. Si no, usa x_left_adjusted.
                # Nota: Estas ya están en el sistema de coordenadas ajustado (m, n, t).
                # Deberían coincidir perfectamente con las cajas que acabamos de ajustar espacialmente.
                x_datos = datos.get("x_left_0_adjusted", datos.get("x_left_adjusted"))[mask_test]
                y_datos = datos.get("y_left_0_adjusted", datos.get("y_left_adjusted"))[mask_test]
                
                # Filtrar NaNs
                valid_mask = ~np.isnan(x_datos) & ~np.isnan(y_datos)
                x_valid = x_datos[valid_mask]
                y_valid = y_datos[valid_mask]
                
                if len(x_valid) == 0:
                    continue
                
                # 3. Mapear coordenadas a Estados
                secuencia_estados = []
                for x, y in zip(x_valid, y_valid):
                    estado_asignado = 0 # Por defecto es fondo
                    for i, bbox in enumerate(cajas_validas):
                        if punto_en_caja(x, y, bbox):
                            estado_asignado = i + 1 # Estado 1, 2, 3...
                            break # Asume que no hay solapamiento o se queda con la primera que toca
                    secuencia_estados.append(estado_asignado)
                
                # 4. Calcular la Matriz de Markov
                matriz_transicion = calcular_matriz_transicion(secuencia_estados, num_estados)
                
                # Guardar resultado
                resultados_markov[fname][img_name] = {
                    'secuencia': secuencia_estados,
                    'matriz': matriz_transicion,
                    'num_cajas': len(cajas_validas)
                }
                
            except Exception as e:
                print(f"Error procesando {img_name} del sujeto {fname}: {e}")
                pass

    # Al finalizar, puedes guardar todos los resultados en un nuevo pickle general
    with open(os.path.join(data_path, 'matrices_markov_todos_sujetos.pkl'), 'wb') as f:
        pickle.dump(resultados_markov, f)
    print("¡Proceso completado y guardado!")

if __name__ == "__main__":
    cadenas_markov_sujetos()