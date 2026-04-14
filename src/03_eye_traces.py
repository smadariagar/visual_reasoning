import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from tqdm import tqdm

# --- RUTAS ---
data_path = '/home/samuel/Documentos/Visual_Reasoning/data/processed/'
save_path = '/home/samuel/Documentos/Visual_Reasoning/data/traces_plots/'

def grafico_verificacion_trial(time_total, x_total, y_total, oc_data_trial, t_fix, t_stim, res_pantalla, fname, trial_idx, img_name):
    
    estilo_fuente = {'family': 'sans-serif', 'size': 12, 'weight': 'bold'}

    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), layout='constrained')
    fig.suptitle(f'Verificación Temporal | Sujeto: {fname} | Trial: {trial_idx} | {img_name}', 
                 fontsize=14, fontweight='bold', fontfamily='sans-serif')

    # 1. Trazado continuo base (gris/negro transparente)
    ax[0].plot(time_total, x_total, color='k', alpha=0.3, linewidth=1, zorder=1)
    ax[1].plot(time_total, y_total, color='k', alpha=0.3, linewidth=1, zorder=1)

    # 2. Extraer posición real de la mirada durante la CRUZ DE FIJACIÓN
    mask_cruz = (time_total >= t_fix) & (time_total < t_stim)
    x_cruz_real = np.nanmedian(x_total[mask_cruz])
    y_cruz_real = np.nanmedian(y_total[mask_cruz])

    # 3. Colorear fijaciones (adaptado de tu código)
    if len(oc_data_trial) > 0:
        num_fijaciones = int(len(oc_data_trial) * 1.5)
        colores = cm.hsv(np.linspace(0, 1, num_fijaciones))

        for index, row in oc_data_trial.iterrows():
            if row['long'] >= 10 and row['is_saccade'] == 0:
                color_actual = colores[index - oc_data_trial.index[0]]
                
                t_s, t_e = row['time_start_ms']/1000.0, row['time_end_ms']/1000.0
                mask_fixation = (time_total >= t_s) & (time_total <= t_e)
                
                ax[0].plot(time_total[mask_fixation], x_total[mask_fixation], color=color_actual, linewidth=2.5, zorder=2)
                ax[1].plot(time_total[mask_fixation], y_total[mask_fixation], color=color_actual, linewidth=2.5, zorder=2)

    # 4. Líneas Diagnósticas
    centro_x_ideal = res_pantalla[0] / 2
    centro_y_ideal = res_pantalla[1] / 2

    for i, eje in enumerate(ax):
        # Transición: Fin Cruz -> Inicio Estímulo
        eje.axvline(t_stim, color='red', linestyle='--', linewidth=2, label='Aparición Imagen' if i==0 else "")
        
    # Eje X: Ideal vs Real
    ax[0].axhline(centro_x_ideal, color='green', linestyle=':', linewidth=2, label=f'Centro Ideal X ({int(centro_x_ideal)})')
    ax[0].axhline(x_cruz_real, color='blue', linestyle='--', linewidth=1.5, label=f'Mirada Real X ({int(x_cruz_real)})')
    
    # Eje Y: Ideal vs Real
    ax[1].axhline(centro_y_ideal, color='green', linestyle=':', linewidth=2, label=f'Centro Ideal Y ({int(centro_y_ideal)})')
    ax[1].axhline(y_cruz_real, color='blue', linestyle='--', linewidth=1.5, label=f'Mirada Real Y ({int(y_cruz_real)})')

    # 5. Formato y Estética (Tu código)
    ax[0].set_ylabel('Posición X (px)', fontdict=estilo_fuente)
    ax[1].set_ylabel('Posición Y (px)', fontdict=estilo_fuente)
    ax[1].set_xlabel('Tiempo (s)', fontdict=estilo_fuente)
    
    # Invertir eje Y para representar la pantalla física
    ax[1].set_ylim(res_pantalla[1], 0)
    
    grosor_linea = 2.0
    for eje in ax:
        eje.tick_params(axis='both', which='major', labelsize=10, width=grosor_linea)
        for borde in eje.spines.values():
            borde.set_linewidth(grosor_linea)
        eje.spines['top'].set_visible(False)
        eje.spines['right'].set_visible(False)
        eje.legend(loc='upper right', fontsize=9)

    # Guardado
    ruta_guardado = os.path.join(save_path, f"{fname}_trial_{trial_idx:03d}.png")
    fig.savefig(ruta_guardado, dpi=300, bbox_inches='tight', transparent=False)    
    plt.close(fig)

# ==========================================
# BUCLE PRINCIPAL
# ==========================================
def recorrer_sujetos():
    os.makedirs(save_path, exist_ok=True)
    
    lista_de_carpetas = [nombre for nombre in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, nombre))]
    lista_de_carpetas.sort() 
    
    for fname in tqdm(lista_de_carpetas, desc="Sujetos (Verificación de Trazos)"):
        file_folder = os.path.join(data_path, fname)
        dat_file = os.path.join(file_folder, fname + '.dat')
        answ_file = os.path.join(file_folder, fname + '_answers.csv')
        comp_oc_file = os.path.join(file_folder, fname + '_oc_events.csv')

        if not os.path.exists(dat_file) or not os.path.exists(answ_file): 
            continue
            
        with open(dat_file, 'rb') as f:
            datos = pickle.load(f)

        res_pantalla = datos.get("screen_resolution", (1920, 1080))
        df_answ = pd.read_csv(answ_file)
        oc_data = pd.read_csv(comp_oc_file)

        for index, row in df_answ.iterrows():
            img_name = row['img_name']
                  
            try:
                # Extraemos el ciclo completo: desde la cruz hasta la respuesta
                t_fix = datos["events"][0][index] / 1000.0
                t_stim = datos["events"][1][index] / 1000.0
                t_resp = datos["events"][2][index] / 1000.0
                
                mask_total = (datos["time_array"] >= t_fix) & (datos["time_array"] <= t_resp)
                
                time_total = datos["time_array"][mask_total]
                x_total = datos["x_left"][mask_total]
                
                # REVERSIÓN MNE: Mantenemos el eje Y como lo graba el hardware (0 arriba)
                y_total = res_pantalla[1] - datos["y_left"][mask_total]

                # Filtrar los eventos de este trial
                mask_oc = (oc_data['time_end_ms']/1000.0 >= t_fix) & (oc_data['time_start_ms']/1000.0 <= t_resp)
                oc_data_trial = oc_data[mask_oc].copy()

                if len(time_total) > 10:
                    grafico_verificacion_trial(
                        time_total, x_total, y_total, oc_data_trial, 
                        t_fix, t_stim, res_pantalla, fname, index, img_name
                    )

            except Exception as e:
                print(f"Error procesando {img_name} en {fname}: {e}")
                continue
                
        break # Quita el break para iterar sobre todos los sujetos

if __name__ == "__main__":
    recorrer_sujetos()