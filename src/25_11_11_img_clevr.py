import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
# Backend 'Agg' es obligatorio para guardar miles de imágenes sin crashear la memoria
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed

# =============================================================================
# 1. CONFIGURACIÓN Y RUTAS
# =============================================================================

PATH_ROOT = '/home/samuelmr/Documentos/Visual Reasoning/'
PATH_IMG_VAL = os.path.join(PATH_ROOT, 'img_question/img_val/') 
PATH_DATA = os.path.join(PATH_ROOT, 'data/data/')                
PATH_JSON = os.path.join(PATH_ROOT, 'CLEVR/CLEVR_v1.0/scenes/CLEVR_val_scenes.json') 
# Carpeta de salida
PATH_OUTPUT = os.path.join(PATH_ROOT, 'results/analisis_completo_final/') 

os.makedirs(PATH_OUTPUT, exist_ok=True)

# Parámetros Transformación
SCALE_FACTOR = 2
X_OFFSET = 480
Y_OFFSET = 220

# Parámetros Yarbus
ROI_RADIUS = 80         
MS_POR_MUESTRA = 2      
MIN_DWELL_TIME_MS = 60  

# =============================================================================
# 2. LÓGICA DE CÁLCULO
# =============================================================================

def get_screen_coords(pixel_coords):
    x = pixel_coords[0] * SCALE_FACTOR + X_OFFSET
    y = pixel_coords[1] * SCALE_FACTOR + Y_OFFSET
    return x, y

def calcular_secuencia_mirada(x_gaze, y_gaze, scene_objects):
    target_coords = []
    target_info = []
    
    for i, obj in enumerate(scene_objects):
        sx, sy = get_screen_coords(obj['pixel_coords'])
        target_coords.append([sx, sy])
        target_info.append({
            'id': i,
            'label': f"{obj['size']} {obj['color']} {obj['shape']}",
            'short_label': f"Obj {i}",
            'x': sx, 'y': sy
        })
    
    if not target_coords:
        return pd.DataFrame(), target_info

    target_coords = np.array(target_coords) 
    gaze_points = np.column_stack((x_gaze, y_gaze)) 
    
    # Vectorización de distancias
    diff = gaze_points[:, np.newaxis, :] - target_coords[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2) 
    
    min_dist_sq = np.min(dist_sq, axis=1)
    nearest_obj_idx = np.argmin(dist_sq, axis=1)
    
    roi_sq = ROI_RADIUS**2
    detected_ids = np.full(len(x_gaze), -1) 
    mask_valid = min_dist_sq <= roi_sq
    detected_ids[mask_valid] = nearest_obj_idx[mask_valid]

    # Compresión RLE (Run-Length Encoding)
    cambios = np.where(detected_ids[:-1] != detected_ids[1:])[0] + 1
    cortes = np.concatenate(([0], cambios, [len(detected_ids)]))
    
    visitas = []
    orden_absoluto = 0

    for i in range(len(cortes) - 1):
        start = cortes[i]
        end = cortes[i+1]
        obj_id = detected_ids[start]
        
        if obj_id != -1:
            duracion = (end - start) * MS_POR_MUESTRA
            if duracion >= MIN_DWELL_TIME_MS:
                orden_absoluto += 1
                obj_data = target_info[obj_id]
                visitas.append({
                    'orden': orden_absoluto,
                    'objeto_id': obj_data['id'],
                    'etiqueta': obj_data['short_label'],
                    'inicio_ms': start * MS_POR_MUESTRA,
                    'duracion_ms': duracion,
                    'x_obj': obj_data['x'],
                    'y_obj': obj_data['y']
                })
                
    return pd.DataFrame(visitas), target_info

# =============================================================================
# 3. GRAFICADORES
# =============================================================================

def graficar_yarbus_semantico(img_path, x_gaze, y_gaze, df_visitas, targets, info_trial, save_path):
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        img = Image.new('RGB', (1920, 1080), color=(173, 173, 173))
    
    fig, ax = plt.subplots(figsize=(16, 9), layout='constrained')
    ax.imshow(np.array(img))
    
    # Trayectoria tenue
    ax.plot(x_gaze, y_gaze, color='white', linewidth=0.8, alpha=0.5)

    # Targets
    for t in targets:
        circ = patches.Circle((t['x'], t['y']), ROI_RADIUS, 
                              linewidth=1, edgecolor='yellow', facecolor='none', linestyle='--', alpha=0.4)
        ax.add_patch(circ)
        ax.plot(t['x'], t['y'], 'rx', markersize=3, alpha=0.6)

    # Etiquetas
    if not df_visitas.empty:
        for obj_id, grupo in df_visitas.groupby('objeto_id'):
            tx = grupo.iloc[0]['x_obj']
            ty = grupo.iloc[0]['y_obj']
            ordenes = [str(o) for o in grupo['orden'].tolist()]
            tiempo_total = grupo['duracion_ms'].sum()
            texto = f"#{','.join(ordenes)}\n{tiempo_total}ms"
            
            ax.text(tx, ty, texto, 
                    color='white', fontsize=9, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.7))

    # Info
    info_str = f"Sujeto: {info_trial['fname']}\nTrial: {info_trial['index']}\nAns: {'Correcta' if info_trial['correct'] else 'Incorrecta'}"
    ax.text(1900, 1060, info_str, color='white', ha='right', va='bottom', fontsize=12,
            bbox=dict(facecolor='black', alpha=0.6))

    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis('off')
    
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def graficar_heatmap(img_path, x_gaze, y_gaze, info_trial, save_path):
    # Limpieza de datos NaN para KDE
    df = pd.DataFrame({'x': x_gaze, 'y': y_gaze}).dropna()
    if df.empty or len(df) < 5: 
        return

    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        img = Image.new('RGB', (1920, 1080), color=(173, 173, 173))

    fig, ax = plt.subplots(figsize=(16, 9), layout='constrained')
    ax.imshow(np.array(img), extent=[0, 1920, 1080, 0])

    try:
        sns.kdeplot(
            data=df, x='x', y='y',
            ax=ax,
            fill=True,
            cmap='viridis',
            alpha=0.5,
            thresh=0.05,
            levels=15,
            bw_adjust=0.6
        )
        if ax.collections:
            clip_box = patches.Rectangle((0, 0), 1920, 1080, transform=ax.transData)
            for collection in ax.collections:
                collection.set_clip_path(clip_box)
                
    except Exception:
        pass

    info_str = f"Sujeto: {info_trial['fname']}\nTrial: {info_trial['index']}\nAns: {'Correcta' if info_trial['correct'] else 'Incorrecta'}"
    ax.text(1900, 1060, info_str, color='white', ha='right', va='bottom', fontsize=12,
            bbox=dict(facecolor='black', alpha=0.6))

    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis('off')
    
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

# =============================================================================
# 4. PROCESAMIENTO CENTRAL (PARALELO)
# =============================================================================

def procesar_un_sujeto(sujeto_folder, clevr_scenes):
    folder_sujeto = os.path.join(PATH_DATA, sujeto_folder)
    path_pickle = os.path.join(folder_sujeto, f"{sujeto_folder}.dat")
    path_csv = os.path.join(folder_sujeto, f"{sujeto_folder}_answers.csv")
    
    if not os.path.exists(path_pickle) or not os.path.exists(path_csv):
        return

    # Cargar Datos
    try:
        with open(path_pickle, 'rb') as f:
            datos_eye = pickle.load(f)
    except Exception:
        return

    # --- AQUÍ ESTÁ LA LÓGICA CRÍTICA DE SELECCIÓN ---
    # Verificamos si existen AMBOS vectores ajustados
    if "x_left_adjusted" in datos_eye and "y_left_adjusted" in datos_eye:
        vector_x = datos_eye["x_left_adjusted"]
        vector_y = datos_eye["y_left_adjusted"]
        # print(f"DEBUG: {sujeto_folder} usando datos AJUSTADOS.") 
    else:
        vector_x = datos_eye["x_left"]
        vector_y = datos_eye["y_left"]
        # print(f"DEBUG: {sujeto_folder} usando datos CRUDOS.")

    df_trials = pd.read_csv(path_csv)
    
    output_sujeto_dir = os.path.join(PATH_OUTPUT, sujeto_folder)
    os.makedirs(output_sujeto_dir, exist_ok=True)

    for idx, row in df_trials.iterrows():
        img_name = row['img_name']
        
        # Mapeo a escena
        try:
            scene_idx = int(img_name.split('_')[0])
            scene_data = clevr_scenes['scenes'][scene_idx]
        except:
            continue

        # Corte temporal
        try:
            t_ini = datos_eye["events"][1][idx]/1000 + 0.2 
            t_fin = datos_eye["events"][2][idx]/1000
            mask = (datos_eye["time_array"] >= t_ini) & (datos_eye["time_array"] <= t_fin)
            
            x_trial = vector_x[mask]
            y_trial = vector_y[mask]
        except:
            continue
        
        if len(x_trial) < 10: continue

        # Información común
        info_trial = {
            'fname': sujeto_folder,
            'img_name': img_name,
            'correct': row.get('correct', 0),
            'index': idx
        }
        
        # Nombres de archivo ordenados
        trial_str = f"{idx:03d}"
        estado = "Correcta" if row.get('correct', 0) else "Incorrecta"
        img_base = img_name.split('.')[0]
        path_img_fondo = os.path.join(PATH_IMG_VAL, img_name)

        # ---------------------------------------------------------
        # A) YARBUS SEMÁNTICO
        # ---------------------------------------------------------
        # df_visitas, targets = calcular_secuencia_mirada(x_trial, y_trial, scene_data['objects'])
        
        # if not df_visitas.empty:
        #     csv_name = f"{sujeto_folder}_{trial_str}_{img_base}_metrics.csv"
        #     df_visitas.to_csv(os.path.join(output_sujeto_dir, csv_name), index=False)

        # save_name_yarbus = f"{sujeto_folder}_{trial_str}_{img_base}_{estado}_analisis.png"
        # graficar_yarbus_semantico(path_img_fondo, x_trial, y_trial, df_visitas, targets, info_trial, 
        #                           os.path.join(output_sujeto_dir, save_name_yarbus))

        # ---------------------------------------------------------
        # B) HEATMAP
        # ---------------------------------------------------------
        save_name_heat = f"{sujeto_folder}_{trial_str}_{img_base}_{estado}_heatmap.png"
        graficar_heatmap(path_img_fondo, x_trial, y_trial, info_trial, 
                         os.path.join(output_sujeto_dir, save_name_heat))

# =============================================================================
# 5. EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    print("--- INICIANDO ANÁLISIS COMPLETO (FINAL) ---")
    
    # Cargar JSON
    try:
        with open(PATH_JSON, 'r') as f:
            clevr_scenes = json.load(f)
    except FileNotFoundError:
        print("ERROR: JSON no encontrado.")
        return

    sujetos = [d for d in os.listdir(PATH_DATA) if os.path.isdir(os.path.join(PATH_DATA, d))]
    sujetos.sort()
    
    print(f"Sujetos: {len(sujetos)} (Procesando en paralelo...)")

    Parallel(n_jobs=-1)(
        delayed(procesar_un_sujeto)(sujeto, clevr_scenes) 
        for sujeto in tqdm(sujetos)
    )

    print(f"\n¡PROCESO TERMINADO!\nResultados en: {PATH_OUTPUT}")

if __name__ == "__main__":
    main()