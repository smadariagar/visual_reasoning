import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mne
from tqdm import tqdm
import pickle
from collections import Counter
import seaborn as sns
import matplotlib.patches as patches

img_path = '/home/samuel/Documentos/Visual_Reasoning/img_question/img_test/'
data_path = '/home/samuel/Documentos/Visual_Reasoning/data/data/'

def crear_heat_map(image_name, xl, yl, xr, yr, save_path=None):
    """
    Crea un mapa de calor (heatmap) del comportamiento ocular sobre una imagen de fondo.

    Args:
        img_path (str): Ruta a la carpeta de imágenes.
        image_name (str): Nombre del archivo de la imagen de fondo.
        xl, yl, xr, yr (list): Listas con las coordenadas X e Y de ambos ojos.
        save_path (str, optional): Ruta para guardar la imagen. Si es None, la muestra.
    """
 
    x_coords = np.nanmean((xl, xr),0)
    y_coords = np.nanmean((yl, yr),0)

    # Es crucial eliminar los valores NaN antes de crear el heatmap
    df = pd.DataFrame({'x': x_coords, 'y': y_coords}).dropna()

    # Si no quedan datos después de limpiar, no se puede generar el mapa.
    if df.empty:
        print(f"Advertencia: No hay datos válidos para generar el heatmap para {image_name}.")
        return

    # 2. Carga la imagen de fondo
    img_original = Image.open(os.path.join(img_path, image_name))
    img_array = np.array(img_original)
    
    # 3. Crea la figura y muestra la imagen de fondo
    fig, ax = plt.subplots(figsize=(9.6, 5.4), layout='constrained')
    ax.set_position([0, 0, 1, 1])
    ax.imshow(img_array, extent=[0, 1920, 1080, 0])

    # 4. Crea y superpone el mapa de calor (KDE Plot) con Seaborn
    sns.kdeplot(
        data=df, x='x', y='y',
        ax=ax,          # Dibuja sobre los ejes existentes
        fill=True,      # Rellena las áreas de densidad
        cmap='viridis',     # Paleta de colores (frío a cálido)
        alpha=0.6,      # Transparencia para ver la imagen de fondo
        thresh=0.02,    # Umbral para no dibujar áreas de muy baja densidad
        levels=10,       # Número de niveles de contorno
        bw_adjust=0.8
    )

    if ax.collections:
        clip_box = patches.Rectangle((0, 0), 1920, 1080, transform=ax.transData)
        ax.collections[-1].set_clip_path(clip_box)

    ax.set_xlim(0,1920)
    ax.set_ylim(1080,0)
    ax.axis('off') # Oculta los ejes para una imagen limpia
    

    # 5. Guarda o muestra la figura
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close(fig) # Libera la memoria


def crear_grafico_fijacion(datos, im, xl, yl, xr, yr, res_path):
    """Crea el gráfico de la cruz de fijación."""
    fig, ax = plt.subplots(figsize=(5, 4.8), layout='constrained')
    ax.set_facecolor('#d3d3d3')
    
    centro_x, centro_y, tamaño_cruz = 1920/2, 1080/2, 50
    ax.plot([centro_x - tamaño_cruz, centro_x + tamaño_cruz], [centro_y, centro_y], color='black', lw=2)
    ax.plot([centro_x, centro_x], [centro_y - tamaño_cruz, centro_y + tamaño_cruz], color='black', lw=2)
    
    ax.set_xlim(centro_x - 300, centro_x + 300)
    ax.set_ylim(centro_y - 300, centro_y + 300)

    mask_cros = (datos["time_array"] >= datos["events"][0][im] / 1000) & \
                (datos["time_array"] <= datos["events"][1][im] / 1000)
    
    ax.plot(datos["x_left_0"][mask_cros[1:]] + xl, datos["y_left_0"][mask_cros[1:]] + yl, color='#FF0000', lw=6)
    ax.plot(datos["x_right_0"][mask_cros[1:]] + xr, datos["y_right_0"][mask_cros[1:]] + yr, color='#2626FF', lw=6)
    
    plt.savefig(res_path)
    plt.close(fig) # MUY IMPORTANTE: cierra la figura para liberar memoria


def crear_grafico_comportamiento(datos, im, xl, yl, xr, yr, res_path):
    """Crea el gráfico del comportamiento ocular en el tiempo."""
    fig, ax = plt.subplots(2, 1, figsize=(8.6, 4.5), layout='constrained', sharex=True)
    fig.suptitle(f'Comportamiento ocular, trial: {im+1}',  fontsize=20)
    
    t_inicio, t_estimulo, t_fin = datos["events"][0][im]/1000, datos["events"][1][im]/1000, datos["events"][2][im]/1000
    mask = (datos["time_array"] >= t_estimulo) & (datos["time_array"] <= t_fin)

    # Graficar en ax[0] (Posición X)
    # ax[0].plot(datos["time_array"][mask], datos["x_left"][mask] + xl, color='#FF0000', alpha=0.2, lw=2, ls='--')
    # ax[0].plot(datos["time_array2"][mask[1:]], datos["x_left_0"][mask[1:]] + xl, color='#FF0000', lw=4, label='Ojo izq')

    # ax[0].plot(datos["time_array"][mask], datos["x_right"][mask] + xr, color='#2626FF', alpha=0.2, lw=2, ls='--')
    # ax[0].plot(datos["time_array2"][mask[1:]], datos["x_right_0"][mask[1:]] + xr, color='#2626FF', lw=4, label='Ojo der')
    ax[0].plot(datos["time_array"][mask], datos["x_left"][mask], color='#1483ff', alpha=0.4, lw=3, ls='--')
    #ax[0].plot(datos["time_array2"][mask[1:]], datos["x_left_0"][mask[1:]], color='#1483ff', lw=4, label='Ojo der')

    # ax[0].axvline(t_estimulo, color='k', ls='-.')
    # ax[0].legend()
    ax[0].set_ylim(0, 1920)
    ax[0].set_xlim(t_estimulo, t_fin)
    ax[0].set_ylabel('Posición X (pix)', fontsize=15)
    ax[0].tick_params(axis='y', labelsize=12)
    
    # Graficar en ax[1] (Posición Y)
    # ax[1].plot(datos["time_array"][mask], 1080 - datos["y_left"][mask] + yl, color='#FF0000', alpha=0.2, ls='--')
    # ax[1].plot(datos["time_array2"][mask[1:]], 1080 - datos["y_left_0"][mask[1:]] + yl, color='#FF0000', lw=2)

    # ax[1].plot(datos["time_array"][mask], 1080 - datos["y_right"][mask] + yr, color='#2626FF', alpha=0.2, ls='--')
    # ax[1].plot(datos["time_array2"][mask[1:]], 1080 - datos["y_right_0"][mask[1:]] + yr, color='#2626FF', lw=2)
    ax[1].plot(datos["time_array"][mask], 1080 - datos["y_left"][mask], color='#1483ff', alpha=0.4, lw=3, ls='--')
    #ax[1].plot(datos["time_array2"][mask[1:]], 1080 - datos["y_left_0"][mask[1:]], color='#1483ff', lw=4)
    
    ax[1].axvline(t_estimulo, color='k', ls='-.')
    # ax[1].text(t_estimulo, -30, 'Inicio\nestímulo', ha='center', va='top', rotation=90)
    ax[1].set_ylim(0, 1080)
    ax[1].set_ylabel('Posición Y (pix)',  fontsize=15)
    ax[1].set_xlabel('Tiempo (s)', fontsize=15)
    ax[1].tick_params(axis='y', labelsize=12)


    for axis in ax:
        # Itera sobre cada uno de los 4 bordes (arriba, abajo, izq, der)
        for spine in axis.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

    plt.savefig(res_path)
    plt.close(fig)


def crear_grafico_imagen(datos, im, img_mod_1, xl, yl, xr, yr, res_path):
    """Crea la imagen con la trayectoria ocular superpuesta."""
    img_original = Image.open(os.path.join(img_path, img_mod_1[im]))
    
    fig, ax = plt.subplots(figsize=(9.6, 5.4), layout='constrained')
    ax.imshow(np.array(img_original))
    
    t_estimulo, t_fin = datos["events"][1][im]/1000, datos["events"][2][im]/1000
    mask_stim = (datos["time_array"] >= t_estimulo) & (datos["time_array"] <= t_fin)
    
    ax.plot(datos["x_left_0"][mask_stim[1:]] + xl, datos["y_left_0"][mask_stim[1:]] + yl, color='#FF0000')
    ax.plot(datos["x_right_0"][mask_stim[1:]] + xr, datos["y_right_0"][mask_stim[1:]] + yr, color='#2626FF')
    ax.axis('off')
    
    plt.savefig(res_path)
    plt.close(fig)


#######################################################################################
################################### Bucle principal ###################################
#######################################################################################
lista_de_carpetas = [nombre for nombre in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, nombre))]
print(f"Las carpetas dentro de '{data_path}' son:")
print(lista_de_carpetas)

subcarpetas = ['fix_cross', 'ocular_behav', 'img_overlay']
image_tot_sj = []
xl, yl, xr, yr = [], [] ,[] ,[]  
cont = 0

for fname in tqdm(lista_de_carpetas, desc="Procesando carpetas"):
    cont = cont+1
    if cont == 2:
        break

    file_folder =  os.path.join(data_path, fname)

    # Si existe el dato lo carga, sino lo crea
    dat_file = os.path.join(file_folder, fname + '.dat')
    if os.path.exists(dat_file):
        print('Existe')
        with open(dat_file, 'rb') as f:
            datos = pickle.load(f)
    else:
        print('No existe')
        break

    all_folders = []
    for subcarpeta in subcarpetas:
        ruta_completa = os.path.join(file_folder, 'results/'+subcarpeta)
        os.makedirs(ruta_completa, exist_ok=True)
        all_folders.append(ruta_completa)
    
    #### Saca los nombres de las imágenes de los archivos dat
    img_mod_1, img_mod_2, img_mod = [], [], []
    for nombre_archivo in os.listdir(file_folder):
        if "MODULO_1" in nombre_archivo:
            with open(os.path.join(file_folder, nombre_archivo), 'r') as f:
                img_mod_1 = [line.strip().strip('"\'') for line in f]
        elif "MODULO_2" in nombre_archivo:
            with open(os.path.join(file_folder, nombre_archivo), 'r') as f:
                img_mod_2 = [line.strip().strip('"\'') for line in f]
    img_mod = img_mod_1+img_mod_2
    image_tot_sj = image_tot_sj+img_mod
    
    im_us = ''
    xl1, yl1, xr1, yr1 = [], [] ,[] ,[]  
    for im in tqdm(range(len(img_mod)), desc=f"Generando gráficos para {fname}", leave=False):
        if img_mod[im] == '0024_000.png':
            if im_us == '':
                im_us = image_tot_sj[im]
            
            t_inicio, t_estimulo = datos["events"][1][im]/1000+0.2, datos["events"][2][im]/1000
            #t_inicio, t_estimulo = datos["events"][1][im]/1000+0.2, np.min([datos["events"][1][im]/1000+1.2, datos["events"][2][im]/1000])

            mask_cros = (datos["time_array"] >= t_inicio) & (datos["time_array"] <= t_estimulo)

            # cont0, cont1 = 0,0
            # ok, ok2 = False, False
            # in_idx, fin_idx = 0, 0
            # for idx, f in enumerate(mask_cros):
            #     if not f:
            #         continue
                                
            #     if (datos["y_left_0"][idx] < 180 or datos["y_right_0"][idx] < 180) and not ok2:
            #         cont0=cont0+1
            #     else:
            #         cont0=0
            #         ok = True

            #     if (cont0 > 5) and ok:
            #         in_idx = idx-5
            #         ok2 = True

            #     if (datos["y_left_0"][idx] > 180 or datos["y_right_0"][idx] > 180):
            #         cont1=cont1+1
            #     else:
            #         cont1=0
                
            #     if (cont1 > 5) and ok2:
            #         fin_idx= idx-8
            #         break
            
            # mask_cros[:in_idx] = False                        
            # mask_cros[fin_idx:] = False


            xl1.extend(datos["x_left"][mask_cros])
            yl1.extend(datos["y_left"][mask_cros])
            xr1.extend(datos["x_right"][mask_cros])
            yr1.extend(datos["y_right"][mask_cros])
        

            namefig = f'{im:03}'+'_'+img_mod[im][:-4]
            crear_grafico_comportamiento(datos, im, 0, 0, 0, 0, os.path.join(all_folders[1], namefig+'_'+subcarpetas[1]+'_test.png'))
            crear_grafico_imagen(datos, im, img_mod, 0, 0, 0, 0, os.path.join(all_folders[2], namefig+'_'+subcarpetas[2]+'_test.png'))
        
            #plt.plot(datos["y_left_0"][mask_cros[1:]])
   
    #crear_heat_map('0069_001.png', xl1, yl1, xr1, yr1, save_path=None)
    #plt.show()
    xl.extend(xl1)
    yl.extend(yl1)
    xr.extend(xr1)
    yr.extend(yr1)
    # img_original = Image.open(os.path.join(img_path, im_us))
    
    # fig, ax = plt.subplots(figsize=(9.6, 5.4), layout='constrained')
    # ax.imshow(np.array(img_original))

    
    # ax.plot(xl, yl, color='#FF0000')
    # ax.plot(xr, yr, color='#2626FF')

    # ax.axis('off')

    if True:
        continue

    for im in tqdm(range(len(img_mod)), desc=f"Generando gráficos para {fname}", leave=False):
        #if im >= 2: continue # Mantén esto para pruebas rápidas
        im = 45
        # Generar las 3 figuras llamando a las funciones
        #xl, yl, xr, yr = 0, 0, 0, 0
        namefig = f'{im:03}'+'_'+img_mod[im][:-4]
        # crear_grafico_fijacion(datos, im, xl, yl, xr, yr, os.path.join(all_folders[0], namefig+'_'+subcarpetas[0]+'.png'))
        crear_grafico_comportamiento(datos, im, 0, 0, 0, 0, os.path.join(all_folders[1], namefig+'_'+subcarpetas[1]+'_test.png'))
        # crear_grafico_imagen(datos, im, img_mod, xl, yl, xr, yr, os.path.join(all_folders[2], namefig+'_'+subcarpetas[2]+'.png'))
        break
        # Calcular corrección de offset (se hace una vez por trial)
        # t_inicio, t_estimulo = datos["events"][0][im]/1000, datos["events"][1][im]/1000
        # mask_cros = (datos["time_array"] >= t_inicio) & (datos["time_array"] <= t_estimulo)
        # xl = 960 - np.nanmedian(datos["x_left_0"][mask_cros[1:]])
        # yl = 540 - np.nanmedian(datos["y_left_0"][mask_cros[1:]])
        # xr = 960 - np.nanmedian(datos["x_right_0"][mask_cros[1:]])
        # yr = 540 - np.nanmedian(datos["y_right_0"][mask_cros[1:]])
               
        # crear_grafico_fijacion(datos, im, xl, yl, xr, yr, os.path.join(all_folders[0], namefig+'_'+subcarpetas[0]+'_adj.png'))
        # crear_grafico_comportamiento(datos, im, im + 1, xl, yl, xr, yr, os.path.join(all_folders[1], namefig+'_'+subcarpetas[1]+'_adj.png'))
        # crear_grafico_imagen(datos, im, img_mod, xl, yl, xr, yr, os.path.join(all_folders[2], namefig+'_'+subcarpetas[2]+'_adj.png'))
    break
# conteo = Counter(image_tot_sj)
# print(conteo.most_common())
# crear_heat_map('0051_000.png', xl, yl, xr, yr, save_path=None)
# plt.show()

print("Proceso completado.")

