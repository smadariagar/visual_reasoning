import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from tqdm import tqdm

# ================= CONFIGURACIÓN =================
# Carpeta donde están los .pkl corregidos (Ground Truth)
CARPETA_PKL = '/home/samuel/Documentos/Visual_Reasoning/CLEVR/img_test_pkl/'

# Carpeta donde están las NUEVAS imágenes generadas (1920x1080 con texto)
CARPETA_IMAGENES_GENERADAS = '/home/samuel/Documentos/Visual_Reasoning/img_question/img_test/'

# Carpeta para guardar el test visual
CARPETA_SALIDA_TEST = '/home/samuel/Documentos/Visual_Reasoning/CLEVR/test_visual_final/'
os.makedirs(CARPETA_SALIDA_TEST, exist_ok=True)

# Parámetros de tu generación de imágenes (para calcular el offset)
ANCHO_CANVAS = 1920
ALTO_CANVAS = 1080
FACTOR_ESCALA = 2
# Asumimos tamaño original CLEVR estándar. Si usas otras, ajusta esto.
ANCHO_ORIGINAL = 480 
ALTO_ORIGINAL = 320

# Calculamos los offsets matemáticos (igual que en tu código de generación)
OFFSET_X = (ANCHO_CANVAS - (ANCHO_ORIGINAL * FACTOR_ESCALA)) // 2
OFFSET_Y = (ALTO_CANVAS - (ALTO_ORIGINAL * FACTOR_ESCALA)) // 2

print(f"Configuración detectada -> Escala: {FACTOR_ESCALA}x | Offset X: {OFFSET_X} | Offset Y: {OFFSET_Y}")

# ================= PROCESO =================

# Listar imágenes generadas (las que tienen nombres tipo 0006_000.png)
imagenes_generadas = [f for f in os.listdir(CARPETA_IMAGENES_GENERADAS) if f.lower().endswith(('.png', '.jpg'))]
imagenes_generadas.sort()

print(f"Procesando {len(imagenes_generadas)} imágenes generadas...")

for nombre_img_gen in tqdm(imagenes_generadas):
    
    # 1. Identificar el ID original para buscar el PKL
    # Tu formato es "0006_000.png". El ID original es "0006".
    try:
        id_str = nombre_img_gen.split('_')[0] # Toma lo que está antes del primer guion bajo
    except:
        print(f"Saltando {nombre_img_gen}, formato desconocido.")
        continue

    # 2. Buscar el archivo PKL correspondiente
    # Buscamos un pkl que termine en "0006.pkl" (ej: CLEVR_val_000006.pkl)
    archivo_pkl = None
    for pkl in os.listdir(CARPETA_PKL):
        if pkl.endswith(f"{id_str}.pkl"):
            archivo_pkl = pkl
            break
    
    if not archivo_pkl:
        # A veces pasa si generaste imágenes que no tienen pkl analizado
        continue 

    # 3. Cargar datos
    ruta_pkl = os.path.join(CARPETA_PKL, archivo_pkl)
    with open(ruta_pkl, 'rb') as f:
        masks_data = pickle.load(f)

    # 4. Cargar la imagen generada (la grande)
    ruta_img_gen = os.path.join(CARPETA_IMAGENES_GENERADAS, nombre_img_gen)
    img = Image.open(ruta_img_gen)
    
    # 5. Plotear
    fig, ax = plt.subplots(figsize=(16, 9)) # Aspecto 16:9
    ax.imshow(np.array(img))
    ax.axis('off')
    
    contador_objetos = 0
    
    for i, objeto in enumerate(masks_data):
        # Solo dibujamos los que marcaste como válidos (human_label == 1)
        if objeto.get('human_label', 0) == 1:
            
            # --- TRANSFORMACIÓN DE COORDENADAS ---
            bbox_orig = objeto['bbox'] # [x, y, w, h] originales
            
            # 1. Escalar
            w_new = bbox_orig[2] * FACTOR_ESCALA
            h_new = bbox_orig[3] * FACTOR_ESCALA
            
            # 2. Trasladar (Offset) + Escalar posición
            x_new = (bbox_orig[0] * FACTOR_ESCALA) + OFFSET_X
            y_new = (bbox_orig[1] * FACTOR_ESCALA) + OFFSET_Y
            
            # -------------------------------------
            
            # Dibujar rectángulo
            rect = patches.Rectangle(
                (x_new, y_new), w_new, h_new, 
                linewidth=2, 
                edgecolor='#00FF00', # Verde flúor
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Etiqueta
            ax.text(
                x_new, y_new - 5, 
                f"ID {i}", 
                color='black', fontsize=9, weight='bold',
                bbox=dict(facecolor='#00FF00', edgecolor='none', alpha=0.8)
            )
            contador_objetos += 1
    
    ax.set_title(f"Test: {nombre_img_gen} (Origen: {archivo_pkl}) | Objetos: {contador_objetos}", fontsize=10)
    
    # Guardar
    plt.savefig(os.path.join(CARPETA_SALIDA_TEST, f"TEST_{nombre_img_gen}"), bbox_inches='tight', dpi=100)
    plt.close(fig)

print(f"\n¡Test completado! Revisa la carpeta: {CARPETA_SALIDA_TEST}")