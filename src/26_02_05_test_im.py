import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- CONFIGURACIÓN ---
# Carpetas de entrada (las mismas de antes)
carpeta_pkl = '/home/samuel/Documentos/Visual_Reasoning/CLEVR/img_test_pkl/'
carpeta_imagenes = '/home/samuel/Documentos/Visual_Reasoning/CLEVR/img_test/'

# Carpeta de salida (donde se guardarán las imágenes finales con todos los recuadros)
carpeta_salida_final = '/home/samuel/Documentos/Visual_Reasoning/CLEVR/final_visualizations/'

# Crear carpeta si no existe
os.makedirs(carpeta_salida_final, exist_ok=True)

# Listar archivos
archivos_pkl = [f for f in os.listdir(carpeta_pkl) if f.endswith('.pkl')]
archivos_pkl.sort()

print(f"Generando visualizaciones finales para {len(archivos_pkl)} imágenes...")

for nombre_pkl in tqdm(archivos_pkl, desc="Guardando imágenes"):
    
    # 1. Cargar datos
    ruta_pkl = os.path.join(carpeta_pkl, nombre_pkl)
    with open(ruta_pkl, 'rb') as f:
        masks_data = pickle.load(f)
    
    # 2. Cargar imagen original
    nombre_imagen = nombre_pkl.replace('.pkl', '.png')
    ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
    
    if not os.path.exists(ruta_imagen):
        continue # Si no está la imagen, saltamos
        
    img = Image.open(ruta_imagen)
    img_np = np.array(img)
    
    # 3. Crear figura
    # DPI alto para que se vea bien la imagen
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_np)
    ax.axis('off') # Quitar ejes para que se vea limpia
    
    objetos_validos = 0
    
    # 4. Dibujar TODOS los bbox válidos
    for i, objeto in enumerate(masks_data):
        
        # --- FILTRO CLAVE ---
        # Verificamos si tiene la etiqueta humana Y si es 1.
        # Usamos .get() por si acaso algún objeto antiguo no tiene la llave (default a 0)
        es_valido = objeto.get('human_label', 0) == 1
        
        if es_valido:
            bbox = objeto['bbox']
            x, y, w, h = bbox
            
            # Dibujar rectángulo (Verde brillante para resaltar)
            rect = patches.Rectangle(
                (x, y), w, h, 
                linewidth=2, 
                edgecolor='#00FF00', # Verde
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Etiqueta pequeña con el ID del objeto
            ax.text(
                x, y - 5, 
                f"ID {i}", 
                color='black', 
                fontsize=8, 
                weight='bold',
                bbox=dict(facecolor='#00FF00', alpha=0.7, edgecolor='none', pad=1)
            )
            objetos_validos += 1
            
    # Título informativo
    ax.set_title(f"Imagen: {nombre_imagen} | Objetos Válidos: {objetos_validos}", fontsize=14)
    
    # 5. Guardar imagen
    nombre_salida = f"VISUAL_{nombre_imagen}"
    ruta_salida = os.path.join(carpeta_salida_final, nombre_salida)
    plt.show()
    # bbox_inches='tight' quita los bordes blancos extra de matplotlib
    plt.savefig(ruta_salida, bbox_inches='tight', pad_inches=0.1)
    
    # Importante: Cerrar la figura para liberar memoria RAM
    plt.close(fig)

print(f"\n¡Listo! Revisa tus imágenes en: {carpeta_salida_final}")