import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ==========================================
# CONFIGURACIÓN DE EDICIÓN
# ==========================================
CARPETA_PKL = '/home/samuelmr/Documentos/Visual Reasoning/CLEVR/img_test_pkl/'
CARPETA_IMG = '/home/samuelmr/Documentos/Visual Reasoning/CLEVR/img_test/'

# ¿Qué archivo y qué objeto quieres arreglar?
TARGET_FILE = 'CLEVR_test_000037.pkl'  # Nombre del archivo
TARGET_ID = 1                       # El ID que sale en la imagen (índice de la lista)

# ==========================================
# CLASE EDITOR DE BBOX
# ==========================================
class BBoxEditor:
    def __init__(self, pkl_path, img_path, obj_id):
        self.pkl_path = pkl_path
        self.obj_id = obj_id
        
        # Cargar datos
        with open(pkl_path, 'rb') as f:
            self.masks_data = pickle.load(f)
            
        # Validar ID
        if obj_id >= len(self.masks_data):
            print(f"Error: El ID {obj_id} no existe. Hay {len(self.masks_data)} objetos.")
            return

        self.obj_data = self.masks_data[obj_id]
        self.bbox = self.obj_data['bbox'] # [x, y, w, h]
        
        # Cargar imagen
        self.img = Image.open(img_path)
        self.img_w, self.img_h = self.img.size
        
        # Configurar figura
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.35) # Espacio para sliders abajo
        
        self.ax.imshow(np.array(self.img))
        self.ax.set_title(f"Editando: {TARGET_FILE} | Objeto ID: {obj_id}", fontsize=14)
        self.ax.axis('off')

        # Dibujar el rectángulo inicial
        self.rect = patches.Rectangle(
            (self.bbox[0], self.bbox[1]), self.bbox[2], self.bbox[3],
            linewidth=2, edgecolor='#00FF00', facecolor='none'
        )
        self.ax.add_patch(self.rect)
        
        # Dibujar otros objetos (en gris tenue para referencia)
        for i, obj in enumerate(self.masks_data):
            if i != obj_id and obj.get('human_label', 0) == 1:
                b = obj['bbox']
                r = patches.Rectangle((b[0], b[1]), b[2], b[3], linewidth=1, edgecolor='gray', facecolor='none', alpha=0.5)
                self.ax.add_patch(r)
                self.ax.text(b[0], b[1]-5, f"ID {i}", color='gray', fontsize=8)

        self.setup_widgets()
        plt.show()

    def setup_widgets(self):
        axcolor = 'lightgoldenrodyellow'
        
        # Posición inicial actual
        x0, y0, w0, h0 = self.bbox
        
        # Sliders
        # Definimos ejes [left, bottom, width, height]
        ax_x = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
        ax_y = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
        ax_w = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
        ax_h = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
        
        # Creamos los sliders con rangos razonables
        # Rango X e Y: 0 al tamaño de la imagen
        # Rango W y H: 1 al tamaño de la imagen
        self.s_x = Slider(ax_x, 'Posición X', 0, self.img_w, valinit=x0, valstep=1)
        self.s_y = Slider(ax_y, 'Posición Y', 0, self.img_h, valinit=y0, valstep=1)
        self.s_w = Slider(ax_w, 'Ancho (W)', 1, self.img_w, valinit=w0, valstep=1)
        self.s_h = Slider(ax_h, 'Alto (H)', 1, self.img_h, valinit=h0, valstep=1)
        
        # Updates
        self.s_x.on_changed(self.update_box)
        self.s_y.on_changed(self.update_box)
        self.s_w.on_changed(self.update_box)
        self.s_h.on_changed(self.update_box)
        
        # Botón Guardar
        ax_save = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.b_save = Button(ax_save, 'Guardar', color='#ccffcc', hovercolor='#99ff99')
        self.b_save.on_clicked(self.save)

    def update_box(self, val):
        # Leer valores de sliders
        x = self.s_x.val
        y = self.s_y.val
        w = self.s_w.val
        h = self.s_h.val
        
        # Actualizar rectángulo visual
        self.rect.set_x(x)
        self.rect.set_y(y)
        self.rect.set_width(w)
        self.rect.set_height(h)
        
        self.fig.canvas.draw_idle()

    def save(self, event):
        # Obtener valores finales
        new_bbox = [int(self.s_x.val), int(self.s_y.val), int(self.s_w.val), int(self.s_h.val)]
        
        print(f"\n--- GUARDANDO CAMBIOS ---")
        print(f"Original: {self.bbox}")
        print(f"Nuevo   : {new_bbox}")
        
        # Actualizar en memoria
        self.masks_data[self.obj_id]['bbox'] = new_bbox
        # Opcional: Actualizar el área también ya que cambiamos el tamaño
        self.masks_data[self.obj_id]['area'] = new_bbox[2] * new_bbox[3]
        
        # Escribir en disco
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(self.masks_data, f)
            
        print(f"Archivo {os.path.basename(self.pkl_path)} actualizado correctamente.")
        plt.close(self.fig)

# ==========================================
# EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    pkl_full_path = os.path.join(CARPETA_PKL, TARGET_FILE)
    
    # Buscamos la imagen asumiendo extensión png, si usas jpg cambia esto
    img_name = TARGET_FILE.replace('.pkl', '.png')
    img_full_path = os.path.join(CARPETA_IMG, img_name)
    
    if os.path.exists(pkl_full_path) and os.path.exists(img_full_path):
        editor = BBoxEditor(pkl_full_path, img_full_path, TARGET_ID)
    else:
        print("Error: No se encuentra el archivo .pkl o la imagen.")
        print(f"Buscando: {pkl_full_path}")