import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
import pickle
from tqdm import tqdm
import random

# --- RUTAS ---
img_path_base = '/home/samuel/Documentos/Visual_Reasoning/img_question/img_test/'
data_path = '/home/samuel/Documentos/Visual_Reasoning/data/processed/'

# ==========================================
# CLASE CALIBRADOR MODIFICADA
# ==========================================
class CalibradorManual:
    def __init__(self, datos, trials_info, sujeto_name, init_params=(1.0, 1.0, 0.0, 0.0, 0.0)):
        self.datos = datos
        self.trials_info = trials_info 
        self.sujeto = sujeto_name
        
        self.init_params = init_params 
        self.a, self.b, self.m, self.n, self.t = init_params
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 9)) 
        plt.subplots_adjust(left=0.05, bottom=0.25, right=0.95, top=0.95) # Más margen inferior
        self.axs = [self.ax]

        self.lines = []     
        self.raw_segments = [] 
        
        self.setup_plots()
        self.setup_widgets()
        
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

        self.raw_segments.append({
            'mask': mask,
            'x_raw': x_seg, 'y_raw': y_seg,
        })
        
        line_raw, = ax.plot(x_seg, y_seg, 'k-', linewidth=1.5, alpha=0.5, label='Mirada')

        self.lines.append({'raw': line_raw})

        if len(x_seg) > 0: ax.plot(x_seg[0], y_seg[0], 'bo', markersize=5)        
        
        ax.axis('off')
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)

    def setup_widgets(self):
        axcolor = 'lightgoldenrodyellow'
        ax_a = plt.axes([0.15, 0.16, 0.3, 0.02], facecolor=axcolor)
        ax_b = plt.axes([0.15, 0.12, 0.3, 0.02], facecolor=axcolor)
        ax_m = plt.axes([0.55, 0.16, 0.3, 0.02], facecolor=axcolor)
        ax_n = plt.axes([0.55, 0.12, 0.3, 0.02], facecolor=axcolor)
        ax_t = plt.axes([0.35, 0.08, 0.3, 0.02], facecolor=axcolor)
        
        self.s_a = Slider(ax_a, 'Escala X', 0.5, 1.5, valinit=self.a, valstep=0.005)
        self.s_b = Slider(ax_b, 'Escala Y', 0.5, 1.5, valinit=self.b, valstep=0.005)
        self.s_m = Slider(ax_m, 'Offset X', -400.0, 400.0, valinit=self.m, valstep=1)
        self.s_n = Slider(ax_n, 'Offset Y', -400.0, 400.0, valinit=self.n, valstep=1)
        self.s_t = Slider(ax_t, 'Rotación (°)', -45.0, 45.0, valinit=self.t, valstep=0.5)
        
        self.s_a.on_changed(self.update_graph)
        self.s_b.on_changed(self.update_graph)
        self.s_m.on_changed(self.update_graph)
        self.s_n.on_changed(self.update_graph)
        self.s_t.on_changed(self.update_graph)
        
        ax_but = plt.axes([0.8, 0.02, 0.15, 0.04]) 
        self.b_next = Button(ax_but, 'Guardar (Enter)', color=axcolor, hovercolor='0.975')
        self.b_next.on_clicked(self.guardar_y_salir)

        ax_reset = plt.axes([0.65, 0.02, 0.1, 0.04])
        self.b_reset = Button(ax_reset, 'Reset', color='#ffcccc', hovercolor='#ff9999')
        self.b_reset.on_clicked(self.reset_values)

    def reset_values(self, event):
        ia, ib, im, in_, it = self.init_params
        self.s_a.set_val(ia)
        self.s_b.set_val(ib)
        self.s_m.set_val(im)
        self.s_n.set_val(in_)
        self.s_t.set_val(it)

    def on_key_press(self, event):
        if event.key == 'enter':
            self.guardar_y_salir(None)

    # --- NUEVA FUNCIÓN MATEMÁTICA ---
    def aplicar_transformacion(self, x, y):
        theta = np.radians(self.t)
        cx, cy = 1920 / 2, 1080 / 2 
        
        x_c = x - cx
        y_c = y - cy
        
        x_s = x_c * self.a
        y_s = y_c * self.b
        
        x_r = x_s * np.cos(theta) - y_s * np.sin(theta)
        y_r = x_s * np.sin(theta) + y_s * np.cos(theta)
        
        return x_r + cx + self.m, y_r + cy + self.n

    def update_graph(self, val):
        self.a = self.s_a.val
        self.b = self.s_b.val
        self.m = self.s_m.val
        self.n = self.s_n.val
        self.t = self.s_t.val
        
        for i, lines_dict in enumerate(self.lines):
            data = self.raw_segments[i]
            
            new_x_raw, new_y_raw = self.aplicar_transformacion(data['x_raw'], data['y_raw'])
            lines_dict['raw'].set_data(new_x_raw, new_y_raw)
            
        self.fig.canvas.draw_idle()

    def guardar_y_salir(self, event):
        print(f"--> Guardado: Offsets ({self.m:.1f}, {self.n:.1f}) | Rot: {self.t:.1f}°")
        for segment in self.raw_segments:
            mask = segment['mask']
            
            x_raw = self.datos["x_left"][mask]
            y_raw = self.datos["y_left"][mask]
            
            x_adj, y_adj = self.aplicar_transformacion(x_raw, y_raw)
            
            self.datos["x_left_adjusted"][mask] = x_adj
            self.datos["y_left_adjusted"][mask] = y_adj

            info = self.trials_info[0]
            if "parametros_muestra" not in self.datos:
                self.datos["parametros_muestra"] = {}
            self.datos["parametros_muestra"][info['row_index']] = (self.a, self.b, self.m, self.n, self.t)
            
        self.finished = True
        plt.close(self.fig)

            
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
        return self.a, self.b, self.m, self.n, self.t

# ==========================================
# BUCLE PRINCIPAL MODIFICADO
# ==========================================
def procesar_sujetos():
    REINICIAR_TODO = False 
    
    lista_de_carpetas = [nombre for nombre in os.listdir(data_path) 
                            if os.path.isdir(os.path.join(data_path, nombre))]
    lista_de_carpetas.sort() 
    
    for fname in tqdm(lista_de_carpetas, desc="Sujetos"):
        print(f"\n--- Procesando Sujeto: {fname} ---")
        
        file_folder = os.path.join(data_path, fname)
        dat_file_original = os.path.join(file_folder, fname + '.dat')
        dat_file_nuevo = os.path.join(file_folder, fname + '_calib_muestra.dat') 
        answ_file = os.path.join(file_folder, fname + '_answers.csv')

        if not os.path.exists(dat_file_original): continue
            
        if os.path.exists(dat_file_nuevo) and not REINICIAR_TODO:
            print("Cargando archivo de muestra existente para continuar...")
            with open(dat_file_nuevo, 'rb') as f:
                datos = pickle.load(f)
        else:
            print("Cargando datos originales por primera vez...")
            with open(dat_file_original, 'rb') as f:
                datos = pickle.load(f)

        if "parametros_muestra" not in datos or REINICIAR_TODO:
            datos["parametros_muestra"] = {}

        if REINICIAR_TODO or "x_left_adjusted" not in datos:
            datos["x_left_adjusted"] = np.copy(datos["x_left"])
            datos["y_left_adjusted"] = np.copy(datos["y_left"])

        df_answ = pd.read_csv(answ_file)
        
        if "trials_aleatorios" not in datos or REINICIAR_TODO:
            todos_los_indices = list(df_answ.index)
            n_trials = min(10, len(todos_los_indices))
            datos["trials_aleatorios"] = random.sample(todos_los_indices, n_trials)
            
            with open(dat_file_nuevo, 'wb') as f: pickle.dump(datos, f)

        trials_to_process = []
        for index, row in df_answ.iterrows():
            if index in datos["trials_aleatorios"]:
                trials_to_process.append({'img_name': row['img_name'], 'row_index': index, 'index': index})

        batch_size = 1 

        for i in range(0, len(trials_to_process), batch_size):
            batch = trials_to_process[i : i + batch_size]
            info_test = batch[0]
            row_idx = info_test['row_index']
            
            if not REINICIAR_TODO and row_idx in datos.get("revision_v2", {}):
                print(f"Saltando trial {row_idx} (ya calibrado en esta muestra)...")
                continue
                
            init_a, init_b, init_m, init_n, init_t = 1.0, 1.0, 0.0, 0.0, 0.0
            
            if not REINICIAR_TODO:
                try:
                    t_ini = datos["events"][1][row_idx]/1000 + 0.2
                    t_end = datos["events"][2][row_idx]/1000
                    mask_test = (datos["time_array"] >= t_ini) & (datos["time_array"] <= t_end)
                    
                    raw_x = datos["x_left"][mask_test]
                    adj_x = datos["x_left_adjusted"][mask_test]
                    raw_y = datos["y_left"][mask_test]
                    adj_y = datos["y_left_adjusted"][mask_test]
                    
                    if len(raw_x) > 0 and not np.array_equal(raw_x, adj_x):
                        valid_x = ~np.isnan(raw_x) & ~np.isnan(adj_x)
                        if np.sum(valid_x) > 10:
                            coef_x = np.polyfit(raw_x[valid_x], adj_x[valid_x], 1)
                            init_a, init_m = coef_x[0], coef_x[1]

                        valid_y = ~np.isnan(raw_y) & ~np.isnan(adj_y)
                        if np.sum(valid_y) > 10:
                            coef_y = np.polyfit(raw_y[valid_y], adj_y[valid_y], 1)
                            init_b, init_n = coef_y[0], coef_y[1]
                except Exception:
                    pass
            
            print(f"Abriendo img {i+1}/10 (Trial real: {row_idx}): {batch[0]['img_name']}")
            
            calibrador = CalibradorManual(datos, batch, fname, init_params=(init_a, init_b, init_m, init_n, init_t))
            _ = calibrador.show()
            
            if not calibrador.finished:
                print("Interrupción. Guardando progreso de la muestra...")
                with open(dat_file_nuevo, 'wb') as f: pickle.dump(datos, f)
                break 
            
            with open(dat_file_nuevo, 'wb') as f: pickle.dump(datos, f)
                
        print(f"Muestra de 10 completada para el sujeto {fname}.")

if __name__ == "__main__":
    procesar_sujetos()