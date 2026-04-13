import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

# =============================================================================
# 1. CONFIGURACIÓN Y CONSTANTES
# =============================================================================

# Rutas (AJUSTA ESTO A TU PC)
PATH_ROOT = '/home/samuelmr/Documentos/Visual Reasoning/'
PATH_IMG_VAL = os.path.join(PATH_ROOT, 'img_question/img_val/') # Imágenes generadas (fondo gris)
PATH_DATA = os.path.join(PATH_ROOT, 'data/data/')                # Datos Eye-Tracker (Sujetos)
PATH_JSON = os.path.join(PATH_ROOT, 'CLEVR/CLEVR_v1.0/scenes/CLEVR_val_scenes.json') # JSON con posiciones
PATH_OUTPUT = os.path.join(PATH_ROOT, 'results/secuencias_objetos/') # Donde guardar resultados

# Parámetros de Transformación (JSON -> Pantalla 1920x1080)
SCALE_FACTOR = 2
X_OFFSET = 480
Y_OFFSET = 220

# Parámetros de Análisis de Mirada
ROI_RADIUS = 80         # Radio en píxeles alrededor del objeto para considerar que lo está mirando
MS_POR_MUESTRA = 2      # Tiempo entre muestras (2ms = 500Hz). Ajusta si tu eyetracker es distinto.
MIN_DWELL_TIME_MS = 60  # Tiempo mínimo para considerar una visita válida (filtro anti-ruido)

# Crear carpeta de salida
os.makedirs(PATH_OUTPUT, exist_ok=True)

# =============================================================================
# 2. FUNCIONES DE LÓGICA
# =============================================================================

def get_screen_coords(pixel_coords):
    """Convierte coordenadas del JSON original a coordenadas de tu pantalla."""
    x = pixel_coords[0] * SCALE_FACTOR + X_OFFSET
    y = pixel_coords[1] * SCALE_FACTOR + Y_OFFSET
    return x, y

def calcular_secuencia_mirada(x_gaze, y_gaze, scene_objects):
    """
    Algoritmo principal: Convierte coordenadas x,y en una secuencia de objetos visitados.
    Retorna: DataFrame con las visitas y la lista de targets con sus coordenadas de pantalla.
    """
    # 1. Preparar objetivos (Targets) en coordenadas de pantalla
    targets = []
    for i, obj in enumerate(scene_objects):
        sx, sy = get_screen_coords(obj['pixel_coords'])
        targets.append({
            'id': i,
            'label': f"{obj['size']} {obj['color']} {obj['material']} {obj['shape']}", # Etiqueta completa
            'short_label': f"Obj {i}", # Etiqueta corta para el gráfico
            'x': sx, 
            'y': sy
        })

    visitas = []
    objeto_actual = None
    inicio_visita_idx = 0
    contador_muestras = 0
    
    # 2. Recorrer la línea de tiempo de la mirada
    for t, (gx, gy) in enumerate(zip(x_gaze, y_gaze)):
        
        # Ignorar parpadeos (NaNs)
        if np.isnan(gx) or np.isnan(gy):
            continue

        obj_detectado = None
        dist_minima = float('inf')

        # Buscar si el ojo está dentro del radio de algún objeto
        for target in targets:
            dist = np.sqrt((gx - target['x'])**2 + (gy - target['y'])**2)
            if dist <= ROI_RADIUS:
                if dist < dist_minima:
                    dist_minima = dist
                    obj_detectado = target
        
        # --- MÁQUINA DE ESTADOS ---
        # Si cambiamos de objeto (o salimos al vacío)
        if obj_detectado != objeto_actual:
            
            # Cerrar la visita anterior si existía
            if objeto_actual is not None:
                duracion = contador_muestras * MS_POR_MUESTRA
                
                # Solo guardamos si supera el tiempo mínimo (filtro de fijación)
                if duracion >= MIN_DWELL_TIME_MS:
                    visitas.append({
                        'orden_temp': len(visitas) + 1, # Orden temporal absoluto
                        'objeto_id': objeto_actual['id'],
                        'etiqueta': objeto_actual['short_label'],
                        'inicio_ms': inicio_visita_idx * MS_POR_MUESTRA,
                        'duracion_ms': duracion,
                        'x_obj': objeto_actual['x'],
                        'y_obj': objeto_actual['y']
                    })
            
            # Iniciar nueva visita
            objeto_actual = obj_detectado
            inicio_visita_idx = t
            contador_muestras = 0
        
        # Si seguimos en el mismo objeto (o en la nada), contamos tiempo
        if objeto_actual is not None:
            contador_muestras += 1

    # Cerrar la última visita pendiente al acabar el trial
    if objeto_actual is not None:
        duracion = contador_muestras * MS_POR_MUESTRA
        if duracion >= MIN_DWELL_TIME_MS:
            visitas.append({
                'orden_temp': len(visitas) + 1,
                'objeto_id': objeto_actual['id'],
                'etiqueta': objeto_actual['short_label'],
                'inicio_ms': inicio_visita_idx * MS_POR_MUESTRA,
                'duracion_ms': duracion,
                'x_obj': objeto_actual['x'],
                'y_obj': objeto_actual['y']
            })

    return pd.DataFrame(visitas), targets

def graficar_resultados(img_path, x_gaze, y_gaze, df_visitas, targets, info_trial, save_path):
    """Genera la imagen final."""
    
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        # Si no encuentra la imagen generada, crear una negra para que no falle
        img = Image.new('RGB', (1920, 1080), color=(173, 173, 173))
        print(f"Aviso: Imagen no encontrada {img_path}, usando fondo gris.")

    fig, ax = plt.subplots(figsize=(16, 9), layout='constrained')
    ax.imshow(np.array(img))
    
    # 1. Dibujar el camino crudo (Yarbus clásico) en gris suave de fondo
    ax.plot(x_gaze, y_gaze, color='white', linewidth=0.5, alpha=0.3)

    # 2. Dibujar las Áreas de Interés (Targets)
    for t in targets:
        # Círculo de la ROI
        circ = patches.Circle((t['x'], t['y']), ROI_RADIUS, 
                              linewidth=1, edgecolor='yellow', facecolor='none', linestyle='--', alpha=0.4)
        ax.add_patch(circ)
        # Centro
        ax.plot(t['x'], t['y'], 'rx', markersize=4, alpha=0.5)

    # 3. Anotar la Secuencia y Tiempos sobre cada objeto
    if not df_visitas.empty:
        # Agrupamos por objeto para resumir la info
        for obj_id, grupo in df_visitas.groupby('objeto_id'):
            # Datos del objeto
            tx = grupo.iloc[0]['x_obj']
            ty = grupo.iloc[0]['y_obj']
            
            # Calcular texto resumen: Orden de visitas y Tiempo total
            ordenes = [str(o) for o in grupo['orden_temp'].tolist()]
            tiempo_total = grupo['duracion_ms'].sum()
            
            texto = f"#{','.join(ordenes)}\n{tiempo_total}ms"
            
            # Dibujar etiqueta
            ax.text(tx, ty, texto, 
                    color='white', fontsize=10, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.7))

    # Info del Sujeto/Trial en la esquina
    info_str = f"Sujeto: {info_trial['fname']}\nImg: {info_trial['img_name']}\nAns: {'Correcta' if info_trial['correct'] else 'Incorrecta'}"
    ax.text(1800, 1000, info_str, color='white', ha='right', va='bottom', 
            bbox=dict(facecolor='black', alpha=0.5))

    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis('off')
    
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

# =============================================================================
# 3. BUCLE PRINCIPAL
# =============================================================================

def main():
    # 1. Cargar JSON de Escenas
    print("Cargando JSON de escenas...")
    try:
        with open(PATH_JSON, 'r') as f:
            clevr_scenes = json.load(f)
    except Exception as e:
        print(f"Error fatal cargando JSON: {e}")
        return

    # 2. Listar Sujetos
    sujetos = [d for d in os.listdir(PATH_DATA) if os.path.isdir(os.path.join(PATH_DATA, d))]
    sujetos.sort()

    for sujeto in tqdm(sujetos, desc="Procesando Sujetos"):
        
        # Rutas del sujeto
        folder_sujeto = os.path.join(PATH_DATA, sujeto)
        path_pickle = os.path.join(folder_sujeto, f"{sujeto}.dat")
        path_csv = os.path.join(folder_sujeto, f"{sujeto}_answers.csv")
        
        if not os.path.exists(path_pickle) or not os.path.exists(path_csv):
            continue

        # Cargar Datos
        with open(path_pickle, 'rb') as f:
            datos_eye = pickle.load(f)
        
        # Preferir datos ajustados si existen (de tu calibración manual)
        if "x_left_adjusted" in datos_eye:
            vector_x = datos_eye["x_left_adjusted"]
            vector_y = datos_eye["y_left_adjusted"]
        else:
            vector_x = datos_eye["x_left"]
            vector_y = datos_eye["y_left"]

        df_trials = pd.read_csv(path_csv)

        # Crear carpeta de salida para este sujeto
        output_sujeto = os.path.join(PATH_OUTPUT, sujeto)
        os.makedirs(output_sujeto, exist_ok=True)

        # 3. Iterar por cada Trial
        for idx, row in df_trials.iterrows():
            img_name = row['img_name']
            
            # Obtener índice de escena desde el nombre de imagen (ej: "000015_000.png" -> 15)
            try:
                scene_idx = int(img_name.split('_')[0])
                scene_data = clevr_scenes['scenes'][scene_idx]
            except:
                # Si falla el nombre o el índice se sale de rango
                continue

            # Obtener tiempos para cortar el vector de mirada
            try:
                t_ini = datos_eye["events"][1][idx]/1000 + 0.2 # +200ms delay estímulo
                t_fin = datos_eye["events"][2][idx]/1000
                
                # Crear máscara de tiempo
                mask = (datos_eye["time_array"] >= t_ini) & (datos_eye["time_array"] <= t_fin)
                
                x_trial = vector_x[mask]
                y_trial = vector_y[mask]
            except Exception as e:
                # Si hay error en índices del pickle
                continue

            if len(x_trial) == 0: continue

            # --- ANÁLISIS ---
            df_visitas, targets = calcular_secuencia_mirada(x_trial, y_trial, scene_data['objects'])
            
            # --- GUARDAR CSV DE MÉTRICAS (Opcional, muy útil para análisis estadístico) ---
            if not df_visitas.empty:
                csv_metrics_name = f"{img_name.split('.')[0]}_metrics.csv"
                df_visitas.to_csv(os.path.join(output_sujeto, csv_metrics_name), index=False)

            # --- GRAFICAR ---
            info_trial = {
                'fname': sujeto,
                'img_name': img_name,
                'correct': row['correct']
            }
            
            full_img_path = os.path.join(PATH_IMG_VAL, img_name)
            save_img_name = f"{img_name.split('.')[0]}_seq.png"
            save_path = os.path.join(output_sujeto, save_img_name)
            
            graficar_resultados(full_img_path, x_trial, y_trial, df_visitas, targets, info_trial, save_path)

    print("¡Proceso Terminado!")

if __name__ == "__main__":
    main()