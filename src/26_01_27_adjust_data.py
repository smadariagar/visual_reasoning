import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
import pickle
from tqdm import tqdm
import math

# --- RUTAS ---
img_path_base = '/home/samuel/Documentos/Visual_Reasoning/img_question/img_test/'
data_path = '/home/samuel/Documentos/Visual_Reasoning/data/data/'

# ==========================================
# CLASE CALIBRADOR (Igual que antes)
# ==========================================
class CalibradorManual:
    def __init__(self, datos, trials_info, sujeto_name, init_params=(1.0, 1.0, 0.0, 0.0)):
        self.datos = datos
        self.trials_info = trials_info 
        self.sujeto = sujeto_name
        # Guardamos los params iniciales también en una variable de respaldo para el Reset
        self.init_params = init_params 
        self.a, self.b, self.m, self.n = init_params
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 9)) 
        plt.subplots_adjust(left=0.05, bottom=0.20, right=0.95, top=0.95)
        self.axs = [self.ax]

        self.lines = []     
        self.raw_segments = [] 
        
        self.setup_plots()
        self.setup_widgets()
        
        # --- MEJORA 1: Conectar Teclado (Tecla Enter para guardar) ---
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.update_graph(None)
        self.finished = False

    def setup_plots(self):
        info = self.trials_info[0]
        ax = self.axs[0]
        self.fig.suptitle(f"Sujeto: {self.sujeto} | Trial: {info['index']} | Img: {info['img_name']}", fontsize=12)
        
        try:
            full_img_path = os.path.join(img_path_base, info['img_name'])
            img_bg = Image.open(full_img_path)
            ax.imshow(np.array(img_bg), extent=[0, 1920, 1080, 0])
        except Exception as e:
            print(f"Error cargando imagen {info['img_name']}: {e}")

        t_inicio = self.datos["events"][1][info['row_index']]/1000 + 0.2
        t_fin = self.datos["events"][2][info['row_index']]/1000
        mask = (self.datos["time_array"] >= t_inicio) & (self.datos["time_array"] <= t_fin)
        
        x_seg = self.datos["x_left"][mask]
        y_seg = self.datos["y_left"][mask]

        try:
            x_seg_0 = self.datos["x_left_0"][mask[1:]] 
            y_seg_0 = self.datos["y_left_0"][mask[1:]]
        except:
            x_seg_0 = self.datos["x_left_0"][mask] 
            y_seg_0 = self.datos["y_left_0"][mask]
        
        self.raw_segments.append({
            'mask': mask,
            'x_raw': x_seg, 'y_raw': y_seg,
            'x_fix': x_seg_0, 'y_fix': y_seg_0
        })
        
        line_raw, = ax.plot(x_seg, y_seg, 'r-', linewidth=1, alpha=0.5, label='Mirada')
        line_fix, = ax.plot(x_seg_0, y_seg_0, 'k-', linewidth=1.5, alpha=0.8, label='Fijaciones')
        self.lines.append({'raw': line_raw, 'fix': line_fix})

        if len(x_seg) > 0: ax.plot(x_seg[0], y_seg[0], 'bo', markersize=5) 
        ax.axis('off')
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)

    def setup_widgets(self):
        axcolor = 'lightgoldenrodyellow'
        ax_a = plt.axes([0.15, 0.12, 0.3, 0.02], facecolor=axcolor)
        ax_b = plt.axes([0.15, 0.08, 0.3, 0.02], facecolor=axcolor)
        ax_m = plt.axes([0.55, 0.12, 0.3, 0.02], facecolor=axcolor)
        ax_n = plt.axes([0.55, 0.08, 0.3, 0.02], facecolor=axcolor)
        
        self.s_a = Slider(ax_a, 'Escala X', 0.8, 1.2, valinit=self.a, valstep=0.005)
        self.s_b = Slider(ax_b, 'Escala Y', 0.8, 1.2, valinit=self.b, valstep=0.005)
        self.s_m = Slider(ax_m, 'Offset X', -400.0, 400.0, valinit=self.m, valstep=1)
        self.s_n = Slider(ax_n, 'Offset Y', -400.0, 400.0, valinit=self.n, valstep=1)
        
        self.s_a.on_changed(self.update_graph)
        self.s_b.on_changed(self.update_graph)
        self.s_m.on_changed(self.update_graph)
        self.s_n.on_changed(self.update_graph)
        
        # Botón Guardar
        ax_but = plt.axes([0.8, 0.02, 0.15, 0.04]) # Un poco más ancho
        self.b_next = Button(ax_but, 'Guardar (Enter)', color=axcolor, hovercolor='0.975')
        self.b_next.on_clicked(self.guardar_y_salir)

        # --- MEJORA 2: Botón Reset ---
        ax_reset = plt.axes([0.65, 0.02, 0.1, 0.04])
        self.b_reset = Button(ax_reset, 'Reset', color='#ffcccc', hovercolor='#ff9999')
        self.b_reset.on_clicked(self.reset_values)

    def reset_values(self, event):
        # Restaura los valores con los que se abrió esta ventana
        ia, ib, im, in_ = self.init_params
        self.s_a.set_val(ia)
        self.s_b.set_val(ib)
        self.s_m.set_val(im)
        self.s_n.set_val(in_)

    def on_key_press(self, event):
        # Si presionas Enter, llama a guardar_y_salir
        if event.key == 'enter':
            self.guardar_y_salir(None)

    def update_graph(self, val):
        self.a = self.s_a.val
        self.b = self.s_b.val
        self.m = self.s_m.val
        self.n = self.s_n.val
        
        for i, lines_dict in enumerate(self.lines):
            data = self.raw_segments[i]
            
            new_x_raw = data['x_raw'] * self.a + self.m
            new_y_raw = data['y_raw'] * self.b + self.n
            lines_dict['raw'].set_data(new_x_raw, new_y_raw)
            
            new_x_fix = data['x_fix'] * self.a + self.m
            new_y_fix = data['y_fix'] * self.b + self.n
            lines_dict['fix'].set_data(new_x_fix, new_y_fix)
            
        self.fig.canvas.draw_idle()

    def guardar_y_salir(self, event):
        print(f"--> Guardado: Offsets ({self.m:.1f}, {self.n:.1f})")
        for segment in self.raw_segments:
            mask = segment['mask']
            
            x_raw = self.datos["x_left"][mask]
            y_raw = self.datos["y_left"][mask]
            
            x_adj = x_raw * self.a + self.m
            y_adj = y_raw * self.b + self.n
            
            self.datos["x_left_adjusted"][mask] = x_adj
            self.datos["y_left_adjusted"][mask] = y_adj

            if "x_left_0_adjusted" in self.datos:
                try:
                    x_fix = self.datos["x_left_0"][mask[1:]]
                    y_fix = self.datos["y_left_0"][mask[1:]]
                    indices_fix = mask[1:]
                except:
                    x_fix = self.datos["x_left_0"][mask]
                    y_fix = self.datos["y_left_0"][mask]
                    indices_fix = mask

                x_fix_adj = x_fix * self.a + self.m
                y_fix_adj = y_fix * self.b + self.n
                self.datos["x_left_0_adjusted"][indices_fix] = x_fix_adj
                self.datos["y_left_0_adjusted"][indices_fix] = y_fix_adj
            
        self.finished = True
        plt.close(self.fig)

    def show(self):
        manager = plt.get_current_fig_manager()
        try:
            manager.window.attributes('-zoomed', True)
        except Exception:
            try:
                manager.full_screen_toggle()
            except Exception:
                pass 
                
        plt.show()
        return self.a, self.b, self.m, self.n

# ==========================================
# BUCLE PRINCIPAL MODIFICADO
# ==========================================

def procesar_sujetos():
    # =======================================================
    # ¡¡¡ CONFIGURACIÓN AQUÍ !!!
    # True = Borra todo lo anterior y empieza de cero.
    # False = Continúa donde lo dejaste (ignora lo ya hecho).
    REINICIAR_TODO = False 
    # =======================================================
    
    lista_de_carpetas = [nombre for nombre in os.listdir(data_path) 
                            if os.path.isdir(os.path.join(data_path, nombre))]
    lista_de_carpetas.sort() 
    
    for fname in tqdm(lista_de_carpetas, desc="Sujetos"):
        print(f"\n--- Procesando Sujeto: {fname} ---")
        
        file_folder = os.path.join(data_path, fname)
        dat_file = os.path.join(file_folder, fname + '.dat')
        answ_file = os.path.join(file_folder, fname + '_answers.csv')

        if not os.path.exists(dat_file): continue
            
        with open(dat_file, 'rb') as f:
            datos = pickle.load(f)

        # --- LÓGICA DE REINICIO ---
        # Si REINICIAR_TODO es True, entra al if y sobreescribe los vectores.
        if REINICIAR_TODO or "x_left_adjusted" not in datos:
            print(">>> RESETEANDO DATOS A ORIGINALES (REINICIO) <<<")
            datos["x_left_adjusted"] = np.copy(datos["x_left"])
            datos["y_left_adjusted"] = np.copy(datos["y_left"])
            datos["x_left_0_adjusted"] = np.copy(datos["x_left_0"])
            datos["y_left_0_adjusted"] = np.copy(datos["y_left_0"])
        else:
            print("Cargando vectores ajustados previos...")

        df_answ = pd.read_csv(answ_file)
        
        trials_to_process = []
        for index, row in df_answ.iterrows():
            trials_to_process.append({'img_name': row['img_name'], 'row_index': index, 'index': index})

        batch_size = 1 
        last_a, last_b, last_m, last_n = 1.0, 1.0, 0.0, 0.0

        for i in range(0, len(trials_to_process), batch_size):
            batch = trials_to_process[i : i + batch_size]
            
            # --- LÓGICA DE SALTO INTELIGENTE ---
            # Solo saltamos si NO estamos en modo reiniciar
            if not REINICIAR_TODO:
                try:
                    info_test = batch[0]
                    t_ini = datos["events"][1][info_test['row_index']]/1000 + 0.2
                    t_end = datos["events"][2][info_test['row_index']]/1000
                    mask_test = (datos["time_array"] >= t_ini) & (datos["time_array"] <= t_end)
                    
                    # 1. Extraemos slices X
                    raw_x = datos["x_left"][mask_test]
                    adj_x = datos["x_left_adjusted"][mask_test]
                    
                    # 2. Extraemos slices Y
                    raw_y = datos["y_left"][mask_test]
                    adj_y = datos["y_left_adjusted"][mask_test]
                    
                    # Verificamos si hubo cambios (si adj != raw)
                    if len(raw_x) > 0 and not np.array_equal(raw_x, adj_x):
                        
                        # --- RECUPERAR A y M (Eje X) ---
                        # Filtramos NaNs para que polyfit no falle
                        valid_x = ~np.isnan(raw_x) & ~np.isnan(adj_x)
                        if np.sum(valid_x) > 10: # Mínimo de puntos para calcular
                            # Ajuste lineal: adj = a * raw + m
                            # polyfit devuelve [pendiente(a), intercepto(m)]
                            coef_x = np.polyfit(raw_x[valid_x], adj_x[valid_x], 1)
                            last_ac = coef_x[0]
                            last_mc = coef_x[1]

                        # --- RECUPERAR B y N (Eje Y) ---
                        valid_y = ~np.isnan(raw_y) & ~np.isnan(adj_y)
                        if np.sum(valid_y) > 10:
                            coef_y = np.polyfit(raw_y[valid_y], adj_y[valid_y], 1)
                            last_bc = coef_y[0]
                            last_nc = coef_y[1]
                        
                        if np.abs(last_ac-1.0) <= 0.01 and np.abs(last_mc) <= 0.01 and np.abs(last_bc-1.0) <= 0.01 and np.abs(last_nc) <= 0.01:
                            1/0

                        elif i % 10 == 0: 
                            print(f"Saltando procesados hasta {i}... (Recuperado: a={last_ac:.2f}, m={last_mc:.1f})")
                        continue 
                except Exception as e: 
                    # Si falla el cálculo matemático, solo imprimimos error y seguimos normal
                    # print(f"Error recuperando params: {e}") 
                    pass
            
            # -----------------------------------
            print(f"Abriendo img {i}: {batch[0]['img_name']}")
            
            calibrador = CalibradorManual(datos, batch, fname, init_params=(last_a, last_b, last_m, last_n))
            last_a, last_b, last_m, last_n = calibrador.show()
            
            if not calibrador.finished:
                print("Interrupción. Guardando...")
                with open(dat_file, 'wb') as f: pickle.dump(datos, f)
                break 
            
            with open(dat_file, 'wb') as f: pickle.dump(datos, f)
                
        print(f"Sujeto {fname} completado.")

if __name__ == "__main__":
    procesar_sujetos()