import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import shutil

# --- CONFIGURACIÓN ---
carpeta_pkl = '/home/samuel/Documentos/Visual_Reasoning/CLEVR/img_test_pkl/'
carpeta_imagenes = '/home/samuel/Documentos/Visual_Reasoning/CLEVR/img_test/'
carpeta_output_imgs = '/home/samuel/Documentos/Visual_Reasoning/CLEVR/bbox_reviews/' # Carpeta para guardar las fotos de los bbox

# Crear carpeta de salida si no existe
os.makedirs(carpeta_output_imgs, exist_ok=True)

# Listar archivos
archivos_pkl = [f for f in os.listdir(carpeta_pkl) if f.endswith('.pkl')]
archivos_pkl.sort()

print(f"Se encontraron {len(archivos_pkl)} archivos para revisar.")
print("---------------------------------------------------------")
print("CONTROLES:")
print("  [y] = YES (Es un objeto válido) -> Guarda 1")
print("  [n] = NO (Es ruido/fondo)       -> Guarda 0")
print("  [q] = QUIT (Salir del programa)")
print("---------------------------------------------------------")

# Variable global para capturar la tecla
user_decision = None

def on_key(event):
    global user_decision
    if event.key == 'y':
        user_decision = 1
    elif event.key == 'n':
        user_decision = 0
    elif event.key == 'q':
        user_decision = 'quit'

# Configurar la figura UNA sola vez afuera para reciclarla (más rápido)
fig, ax = plt.subplots(figsize=(10, 6))
# Conectar el evento del teclado
fig.canvas.mpl_connect('key_press_event', on_key)

stop_program = False
c = -1
for nombre_pkl in archivos_pkl:
    c=c+1

    if c != 28:
        continue
    
    if stop_program: break

    # 1. Cargar datos
    ruta_pkl = os.path.join(carpeta_pkl, nombre_pkl)
    with open(ruta_pkl, 'rb') as f:
        masks_data = pickle.load(f)
    
    # 2. Cargar imagen
    nombre_imagen = nombre_pkl.replace('.pkl', '.png')
    ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
    
    if not os.path.exists(ruta_imagen):
        continue
        
    img = Image.open(ruta_imagen)
    img_np = np.array(img)
    
    # Bandera para saber si modificamos algo en este archivo
    archivo_modificado = False
    
    # 3. Iterar sobre objetos
    # Usamos enumerate para tener el índice
    for i, objeto in enumerate(masks_data):

        
        # --- FILTROS AUTOMÁTICOS (Los que tú definiste) ---
        d_pred_iou = 0.005
        # Si no pasa tus filtros, lo marcamos como 0 automáticamente o lo saltamos?
        # Aquí asumo que si el algoritmo lo descarta, le ponemos etiqueta -1 (ignorado) o 0.
        # Para tu caso, simplemente haremos 'continue' pero NO le pediremos opinión al humano.
        
        skip = False
        # if objeto['area'] >= 50000: skip = True
        # if objeto['predicted_iou'] + d_pred_iou < 1: skip = True
        
        if skip:
            # Opcional: Marcar como descartado auto
            objeto['human_label'] = 0 
            archivo_modificado = True # Para guardar que fue descartado
            continue

        # --- SI PASA EL FILTRO, PREGUNTAR AL HUMANO ---
        
        # Limpiar gráfico anterior
        ax.clear()
        ax.imshow(img_np)
        ax.axis('off')
        
        # Dibujar bbox
        bbox = objeto['bbox']
        x, y, w, h = bbox
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='#00FF00', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"Img: {nombre_imagen} | Obj ID: {i}\n¿Es un objeto? (y/n)", fontsize=12, color='blue')
        
        # Forzar dibujado
        plt.draw()
        plt.show()
        # --- BUCLE DE ESPERA DE TECLA ---
        user_decision = None
        while user_decision is None:
            # Pausa pequeña para permitir que la GUI responda a eventos
            # Esto mantiene la ventana viva esperando el evento 'on_key'
            plt.pause(0.1) 
        
        # Procesar decisión
        if user_decision == 'quit':
            print("Saliendo del programa...")
            stop_program = True
            break
        
        # Guardar la etiqueta en el diccionario del objeto
        objeto['human_label'] = user_decision
        archivo_modificado = True
        
        etiqueta_str = "OBJETO" if user_decision == 1 else "RUIDO"
        print(f" -> {nombre_imagen} [Obj {i}]: Marcado como {etiqueta_str}")
        
        # --- GUARDAR LA FOTO DE EVIDENCIA ---
        # Nombre: nombreimagen_obj_ID_label.png
        nombre_foto_out = f"{os.path.splitext(nombre_imagen)[0]}_obj_{i}_label_{user_decision}.png"
        ruta_foto_out = os.path.join(carpeta_output_imgs, nombre_foto_out)
        
        # Guardamos lo que se ve en pantalla (crop visual)
        fig.savefig(ruta_foto_out)

    # 4. SOBRESCRIBIR EL PICKLE CON LA NUEVA INFO
    # Solo si hubo cambios o evaluaciones en esta imagen
    if archivo_modificado:
        with open(ruta_pkl, 'wb') as f:
            pickle.dump(masks_data, f)
        print(f" [GUARDADO] Archivo actualizado: {nombre_pkl}")

plt.close(fig)
print("Proceso finalizado.")