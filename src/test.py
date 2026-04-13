import os
import csv
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
from scipy.stats import mannwhitneyu, sem, pearsonr

img_path = '/home/samuelmr/Documentos/Visual Reasoning/img_question/img_test/'
data_path = '/home/samuelmr/Documentos/Visual Reasoning/data/data/'

def histogram_eye(x, y, info):
        
    if len(x)==0 or len(y)==0:
        return
    
    fig, ax = plt.subplots(figsize=(9.6, 5.4), layout='constrained')

    
    ax.hist(y,bins=100)


    # ax.set_xlim(0,1920)
    # ax.set_ylim(1080,0)

    estado = "Correcta" if info['es_correcta'] else "Incorrecta"
    tiempo_ms = int(info['tiempo_total'] * 1000)

    img_base_name = os.path.splitext(info['img_name'])[0]
    texto_info = (
        f"Imagen: {img_base_name}\n"
        f"Sujeto: {info['fname']}"
    )

    final_filename = f"{img_base_name}_{info['fname']}"

    if int(info['grupo']) == 0:
        texto_info = ( 
            f"{texto_info}\n"
            f"Trial: {info['index']}\n"
            f"Respuesta: {estado}\n"
            f"Duración trial: {tiempo_ms} ms"
        )

        final_filename = f"{final_filename}_{info['index']}_{info['es_correcta']}_{tiempo_ms}"

    final_filename = f"{final_filename}.png"


    props = dict(boxstyle='round', facecolor='black', alpha=0.6)

    # c) Añade el texto al gráfico
    ax.text(
        1510, 1050,     # Coordenadas (x, y) - Cerca de la esquina (1920, 1080)
        texto_info,
        fontsize=10,
        color='white',
        ha='left',     # Alineación horizontal: derecha
        va='bottom',    # Alineación vertical: abajo
        bbox=props      # Aplica el cuadro de texto
    )


    base_results_path = '/home/samuelmr/Documentos/Visual Reasoning/results/results_yarbus'
    
    target_folder_path = os.path.join(base_results_path, img_base_name)
    os.makedirs(target_folder_path, exist_ok=True)
    
    save_path = os.path.join(target_folder_path, final_filename)
    #plt.savefig(save_path)
    plt.show()


def crear_heat_map(x, y, info):
    """
    Crea un mapa de calor (heatmap) del comportamiento ocular sobre una imagen de fondo.

    Args:
        img_path (str): Ruta a la carpeta de imágenes.
        image_name (str): Nombre del archivo de la imagen de fondo.
        xl, yl, xr, yr (list): Listas con las coordenadas X e Y de ambos ojos.
        save_path (str, optional): Ruta para guardar la imagen. Si es None, la muestra.
    """

    # Es crucial eliminar los valores NaN antes de crear el heatmap
    df = pd.DataFrame({'x': x, 'y': y}).dropna()

    # Si no quedan datos después de limpiar, no se puede generar el mapa.
    if df.empty:
        print(f"Advertencia: No hay datos válidos para generar el heatmap para {info['img_name']}.")
        return

    # 2. Carga la imagen de fondo
    img_original = Image.open(os.path.join('/home/samuelmr/Documentos/Visual Reasoning/img_question/img_test/', info['img_name']))
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
        bw_adjust=0.5
    )

    if ax.collections:
        clip_box = patches.Rectangle((0, 0), 1920, 1080, transform=ax.transData)
        ax.collections[-1].set_clip_path(clip_box)

    ax.set_xlim(0,1920)
    ax.set_ylim(1080,0)
    ax.axis('off') 

    img_base_name = os.path.splitext(info['img_name'])[0]
    texto_info = (
        f"Imagen: {img_base_name}\n"
        f"Sujeto: {info['fname']}"
    )

    estado = "Correcta" if info['es_correcta'] else "Incorrecta"
    tiempo_ms = int(info['tiempo_total'] * 1000)
    final_filename = f"{img_base_name}_{info['fname']}"
    if int(info['grupo']) == 0:
        texto_info = ( 
            f"{texto_info}\n"
            f"Trial: {info['index']}\n"
            f"Respuesta: {estado}\n"
            f"Duración trial: {tiempo_ms} ms"
        )
        final_filename = f"{final_filename}_{info['index']}_{info['es_correcta']}_{tiempo_ms}"
    final_filename = f"{final_filename}.png"
        
    
    props = dict(boxstyle='round', facecolor='black', alpha=0.6)
    ax.text(
        1510, 1050,     # Coordenadas (x, y) - Cerca de la esquina (1920, 1080)
        texto_info,
        fontsize=10,
        color='white',
        ha='left',     # Alineación horizontal: derecha
        va='bottom',    # Alineación vertical: abajo
        bbox=props      # Aplica el cuadro de texto
    )

    base_results_path = '/home/samuelmr/Documentos/Visual Reasoning/results/heat_maps'

    target_folder_path = os.path.join(base_results_path, img_base_name)
    os.makedirs(target_folder_path, exist_ok=True)
    
    save_path = os.path.join(target_folder_path, final_filename)
    plt.savefig(save_path)
    plt.close(fig)

#######################################################################################
################################### Bucle principal ###################################
#######################################################################################

lista_de_carpetas = [nombre for nombre in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, nombre))]
print(f"Las carpetas dentro de '{data_path}' son:")
print(lista_de_carpetas)

df_corr_anw = pd.read_csv('/home/samuelmr/Documentos/Visual Reasoning/correct_answers.csv')

df_total_answ = pd.read_csv('/home/samuelmr/Documentos/Visual Reasoning/datos_totales_con_conteo.csv')
df_total_corr_answ = pd.read_csv('/home/samuelmr/Documentos/Visual Reasoning/datos_correctos_con_conteo.csv')

#-----
valor_maximo = df_total_answ['Total'].max()
df_filtrado = df_total_answ[df_total_answ['Total'] == valor_maximo]

lista_imagenes_objetivo = df_filtrado['img_name'].tolist()
set_imagenes_objetivo = set(lista_imagenes_objetivo)

cont = 0
df_corr, df_incorr = [], []
for fname in tqdm(lista_de_carpetas, desc="Procesando carpetas"):
    # if cont == 0:
    #     break

    file_folder =  os.path.join(data_path, fname)
    answ_file = os.path.join(file_folder, fname + '_answers.csv')

    df_answ = pd.read_csv(answ_file)

    # Si existe el dato lo carga, sino lo crea
    dat_file = os.path.join(file_folder, fname + '.dat')

    if os.path.exists(dat_file):
        print('Existe')
        with open(dat_file, 'rb') as f:
            datos = pickle.load(f)
    else:
        print('No existe!!!')
        continue

    if True:
        for index, row in df_answ.iterrows():        
            
            if row['img_name'] in set_imagenes_objetivo:
                print(row['img_name'])
                
                t_inicio, t_fin = datos["events"][1][index]/1000+0.2, datos["events"][2][index]/1000
                mask_stim = (datos["time_array"] >= t_inicio) & (datos["time_array"] <= t_fin)

                info_trial = {
                    'img_name': row['img_name'],
                    'index': index,
                    'fname': fname,
                    'es_correcta': row['correct'],
                    'tiempo_total': row['time_trial'],
                    'grupo': 0}

                histogram_eye(datos["x_left"][mask_stim], datos["y_left"][mask_stim], info_trial)


print("Proceso completado.")

