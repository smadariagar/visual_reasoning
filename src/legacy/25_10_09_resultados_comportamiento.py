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

img_path = '/home/samuel/Documentos/Visual_Reasoning/img_question/img_test/'
data_path = '/home/samuel/Documentos/Visual_Reasoning/data/data/'
df_corr_anw = pd.read_csv('/home/samuel/Documentos/Visual_Reasoning/correct_answers.csv')

def time_on_image(datos, mask):
    xl = datos["x_left_0"][mask[1:]]
    yl = datos["y_left_0"][mask[1:]]
    xr = datos["x_right_0"][mask[1:]]
    yr = datos["y_right_0"][mask[1:]]

    toil = (np.sum(yl>200)*2)/1000
    toir = (np.sum(yr>200)*2)/1000

    return np.mean([toil, toir])


#######################################################################################
################################### Bucle principal ###################################
#######################################################################################
lista_de_carpetas = [nombre for nombre in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, nombre))]
print(f"Las carpetas dentro de '{data_path}' son:")
print(lista_de_carpetas)

df_corr_anw = pd.read_csv('/home/samuel/Documentos/Visual_Reasoning/correct_answers.csv')

cont = 0
for fname in tqdm(lista_de_carpetas, desc="Procesando carpetas"):
    # cont = cont+1
    # if cont == 2:
    #     break

    file_folder =  os.path.join(data_path, fname)

    # Si existe el dato lo carga, sino lo crea
    dat_file = os.path.join(file_folder, fname + '.dat')
    if os.path.exists(dat_file):
        print('Existe')
        with open(dat_file, 'rb') as f:
            datos = pickle.load(f)
    else:
        print('No existe!!!')

    csv_name = os.path.join(file_folder, fname+'_answers.csv')

    trial_long = (datos["events"][2] - datos["events"][1])/1000
    fix_c_long = datos["events"][1] - datos["events"][0]

    with open(csv_name, 'w', newline='', encoding='utf-8') as archivo_csv:

        escritor_csv = csv.writer(archivo_csv)
        encabezado = ['img_name','type_of_question','correct_answer','answers','correct','time_trial','time_img','long_question','words']
        escritor_csv.writerow(encabezado)

        for index, image in enumerate(datos["images_list"]):

            fila_encontrada = df_corr_anw[df_corr_anw['img_name'] == image]
            if not fila_encontrada.empty:
                c_a = fila_encontrada['correct_answer'].iloc[0]
                t_q = fila_encontrada['type_of_question'].iloc[0]
                l_q = fila_encontrada['long_question'].iloc[0]
                c_w = fila_encontrada['words'].iloc[0]

            corr_a = 0
            if c_a == datos["responses"][index]:
                corr_a = 1

            t_inicio, t_estimulo = datos["events"][1][index]/1000, datos["events"][2][index]/1000
            mask = (datos["time_array"] >= t_inicio) & (datos["time_array"] <= t_estimulo)

            toi = time_on_image(datos, mask)

            fila = [image, t_q, c_a, datos["responses"][index], corr_a, trial_long[index], toi, l_q, c_w]
            escritor_csv.writerow(fila)


print("Proceso completado.")

