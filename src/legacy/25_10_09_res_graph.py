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

img_path = '/home/samuel/Documentos/Visual_Reasoning/img_question/img_test/'
data_path = '/home/samuel/Documentos/Visual_Reasoning/data/data/'
result_path = '/home/samuel/Documentos/Visual_Reasoning/results/'

def g_barras_totales(df_corr, df_incorr):

    colors = ['#00FF7D','#FF0700']
    x_T = ['Correctas', 'Incorrectas']

    # -----------------
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    fig.suptitle(f'Cantidad respuestas correctas/incorrectas\nTodos los sujetos',  fontsize=16)

    ax.bar(x_T, [len(df_corr), len(df_incorr)], color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Frecuencia', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    fig.savefig(result_path+'barras_totales_1.png')

    # -----------------
    tot_a = np.sum([len(df_corr), len(df_incorr)])
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    fig.suptitle(f'Porcentaje respuestas correctas/incorrectas\nTodos los sujetos',  fontsize=16)
    ax.bar(x_T, [len(df_corr), len(df_incorr)]/tot_a*100, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_ylim(0, 100)
    ax.set_ylabel('Porcentaje', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    fig.savefig(result_path+'barras_totales_2.png')

    # -----------------
    c_suj = df_corr['suj_id'].max()
    suj_corr, suj_incorr = [], []
    for i in range(c_suj+1):
        aux_c = len(df_corr[df_corr['suj_id'] == i])
        aux_i = len(df_incorr[df_incorr['suj_id'] == i])

        suj_corr.append(aux_c/(aux_c+aux_i)*100)
        suj_incorr.append(aux_i/(aux_c+aux_i)*100)

    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    fig.suptitle(f'Distribucion relación correctas/incorrectas\nTodos los sujetos',  fontsize=16)
    
    ax.bar(x_T, [np.mean(suj_corr), np.mean(suj_incorr)], color=colors, edgecolor='black', linewidth=1.5)
    ax.errorbar(x_T, [np.mean(suj_corr), np.mean(suj_incorr)], [np.std(suj_corr), np.std(suj_incorr)], fmt='none', ecolor='k', linewidth=2, capsize=10) 
    ax.set_ylim(0, 100) 
    ax.set_ylabel('Porcentaje', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    stat, p_value = mannwhitneyu(suj_corr, suj_incorr, alternative='two-sided')
    alpha = 0.05  
    if p_value < alpha:
        means = [np.mean(suj_corr), np.mean(suj_incorr)]
        errors = [np.std(suj_corr), np.std(suj_incorr)]
        x1, x2 = 0, 1   
        y = max(means[0] + errors[0], means[1] + errors[1]) + 4
        altura_pata = 2

        ax.plot([x1, x2], [y, y], lw=1.5, c='k')
        ax.plot([x1, x1], [y - altura_pata, y], lw=1.5, c='k')
        ax.plot([x2, x2], [y - altura_pata, y], lw=1.5, c='k')
        ax.text((x1 + x2) * 0.5, y, "*", ha='center', va='bottom', color='k', fontsize=12)

    fig.savefig(result_path+'barras_totales_3.png')
    plt.close('all')
    
def g_barras_totales_grupos(df_corr, df_incorr, names):

    suj_corr, suj_incorr = [], []
    suj_corr_l, suj_incorr_l = [], []
    for i in df_corr['suj_id'].unique():
        aux_c = len(df_corr[df_corr['suj_id'] == i])
        aux_i = len(df_incorr[df_incorr['suj_id'] == i])

        suj_corr_l.append(aux_c)
        suj_incorr_l.append(aux_i)

        suj_corr.append(aux_c/(aux_c+aux_i)*100)
        suj_incorr.append(aux_i/(aux_c+aux_i)*100)
    
    #-----------------
    b = [i for i in range(1, 34) if i % 3 != 0]     
    posiciones_ticks = [(b[i] + b[i+1]) / 2 for i in range(0, len(b), 2)]   

    base_colors = ['#00FF7D', '#FF0700']
    colors = base_colors * 11

    lista1 = suj_corr_l
    lista2 = suj_incorr_l
    tot = [elemento for par in zip(lista1, lista2) for elemento in par]
    #print(tot)

    fig, ax = plt.subplots(figsize=(15, 5), layout='constrained')
    fig.suptitle(f'Cantidad de respuestas correctas/incorrectas por sujeto',  fontsize=16)

    ax.bar(b, tot, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(posiciones_ticks)
    ax.set_xticklabels(names)
    ax.set_ylabel('Frecuencia', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    fig.savefig(result_path+'barras_totales_grupos_1.png')


    #-----------------
    lista1 = suj_corr
    lista2 = suj_incorr
    tot = [elemento for par in zip(lista1, lista2) for elemento in par]
    #print(tot)

    fig, ax = plt.subplots(figsize=(15,5), layout='constrained')
    fig.suptitle(f'Porcentaje de respuestas correctas/incorrectas por sujeto',  fontsize=16)

    ax.bar(b, tot, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(posiciones_ticks)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Porcentaje', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    fig.savefig(result_path+'barras_totales_grupos_2.png')
    plt.close('all')

def g_barras_tipos_grupos(df_corr, df_incorr):

    suj_corr, suj_incorr = [], []
    suj_corr_l, suj_incorr_l = [], []
    for i in df_corr['type_of_question'].unique():
        print(i)
        aux_c = len(df_corr[df_corr['type_of_question'] == i])
        aux_i = len(df_incorr[df_incorr['type_of_question'] == i])

        suj_corr_l.append(aux_c)
        suj_incorr_l.append(aux_i)

        suj_corr.append(aux_c/(aux_c+aux_i)*100)
        suj_incorr.append(aux_i/(aux_c+aux_i)*100)

    b = [1,2,4,5,7,8,10,11]
    posiciones_ticks = [1.5, 4.5, 7.5, 10.5]
    names = ['Si o No', 'Cantidad', 'Tamaño', 'Material']

    colors = ['#00FF7D','#FF0700','#00FF7D','#FF0700','#00FF7D','#FF0700','#00FF7D','#FF0700','#00FF7D','#FF0700']

    lista1 = suj_corr_l
    lista2 = suj_incorr_l
    tot = [elemento for par in zip(lista1, lista2) for elemento in par]
    #print(tot)

    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    fig.suptitle(f'Cantidad de respuestas correctas/incorrectas por tipo de pregunta',  fontsize=16)

    ax.bar(b, tot, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xticks(posiciones_ticks)
    ax.set_xticklabels(names)
    ax.set_ylabel('Frecuencia', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)   

    lista1 = suj_corr
    lista2 = suj_incorr
    tot = [elemento for par in zip(lista1, lista2) for elemento in par]
    #print(tot)
    fig.savefig(result_path+'barras_tipos_grupos_1.png')

    #---------------
    fig, ax = plt.subplots(figsize=(13,5), layout='constrained')
    fig.suptitle(f'Porcentaje de respuestas correctas/incorrectas por tipo de pregunta',  fontsize=16)

    ax.bar(b, tot, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(posiciones_ticks)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Porcentaje', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    fig.savefig(result_path+'barras_tipos_grupos_2.png')
    plt.close('all')

def g_hist_tiempos_grupos(df_corr, df_incorr):

    colors = ['#00FF7D','#FF0700']
    #x_T = ['Correctas', 'Incorrectas']

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')

    ax[0].hist(df_corr['time_trial'].tolist(), bins=range(0,50,5), facecolor=colors[0],edgecolor='black', linewidth=1.5)

    ax[1].hist(df_incorr['time_trial'].tolist(), bins=range(0,50,5), facecolor=colors[1],edgecolor='black', linewidth=1.5)

    ax[0].set_title(f'Histograma tiempos de respuesta correctas\nTodos los sujetos',  fontsize=16)
    ax[1].set_title(f'Histograma tiempos de respuesta incorrectas\nTodos los sujetos',  fontsize=16)
    ax[0].set_ylabel('Frecuencia', fontsize=14)
    ax[0].set_xlabel('Tiempo [s]', fontsize=14)
    ax[1].set_xlabel('Tiempo [s]', fontsize=14)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)

    fig.savefig(result_path+'hist_tiempos_grupos_1.png')
    plt.close('all')

def g_barras_tiempos(df_corr, df_incorr):
      
    colors = ['#00FF7D','#FF0700']
    x_T = ['Correctas', 'Incorrectas']

    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    fig.suptitle(f'Promedio de tiempo de reacción\nTodos los sujetos',  fontsize=16)

    ax.bar(x_T, [df_corr['time_trial'].mean(), df_incorr['time_trial'].mean()], color=colors, edgecolor='black', linewidth=1.5)

    ax.errorbar(x_T, [df_corr['time_trial'].mean(), df_incorr['time_trial'].mean()], [df_corr['time_trial'].std(), df_incorr['time_trial'].std()], fmt='none', ecolor='k', capsize=10, linewidth=2) 

    ax.set_ylabel('Tiempo [s]', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    means = [df_corr['time_trial'].mean(), df_incorr['time_trial'].mean()]
    errors = [df_corr['time_trial'].std(), df_incorr['time_trial'].std()]
    x1, x2 = 0, 1 
    
    stat, p_value = mannwhitneyu(df_corr['time_trial'], df_incorr['time_trial'], alternative='two-sided')
    alpha = 0.05  
    if p_value < alpha: 
        y = max(means[0] + errors[0], means[1] + errors[1]) +1
        altura_pata = 0.5

        ax.plot([x1, x2], [y, y], lw=1.5, c='k')
        ax.plot([x1, x1], [y - altura_pata, y], lw=1.5, c='k')
        ax.plot([x2, x2], [y - altura_pata, y], lw=1.5, c='k')
        ax.text((x1 + x2) * 0.5, y, "*", ha='center', va='bottom', color='k', fontsize=12)

    fig.savefig(result_path+'barras_tiempos_1.png')
    plt.close('all')

def g_barras_dificultad(df_corr, df_incorr):
    colors = ['#00FF7D','#FF0700']
    x_T = ['Correctas', 'Incorrectas']

    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    fig.suptitle(f'Promedio largo de la pregunta\nTodos los sujetos',  fontsize=16)

    ax.bar(x_T, [df_corr['long_question'].mean(), df_incorr['long_question'].mean()], color=colors, edgecolor='black', linewidth=1.5)

    ax.errorbar(x_T, [df_corr['long_question'].mean(), df_incorr['long_question'].mean()], [df_corr['long_question'].std(), df_incorr['long_question'].std()], fmt='none', ecolor='k', capsize=10, linewidth=2) 

    ax.set_ylabel('Frecuencia', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    #ax.set_ylim(0, 100) 

    means = [df_corr['long_question'].mean(), df_incorr['long_question'].mean()]
    errors = [df_corr['long_question'].std(), df_incorr['long_question'].std()]
    x1, x2 = 0, 1 

    stat, p_value = mannwhitneyu(df_corr['long_question'], df_incorr['long_question'], alternative='two-sided')
    alpha = 0.05  
    if p_value < alpha:
        y = max(means[0] + errors[0], means[1] + errors[1])+4
        altura_pata = 2

        ax.plot([x1, x2], [y, y], lw=1.5, c='k')
        ax.plot([x1, x1], [y - altura_pata, y], lw=1.5, c='k')
        ax.plot([x2, x2], [y - altura_pata, y], lw=1.5, c='k')
        ax.text((x1 + x2) * 0.5, y, "*", ha='center', va='bottom', color='k', fontsize=12)

    fig.savefig(result_path+'barras_dificultad_1.png')

    #-----------------------
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    fig.suptitle(f'Promedio cantidad de palabras\nTodos los sujetos',  fontsize=16)

    ax.bar(x_T, [df_corr['words'].mean(), df_incorr['words'].mean()], color=colors, edgecolor='black', linewidth=1.5)

    ax.errorbar(x_T, [df_corr['words'].mean(), df_incorr['words'].mean()], [df_corr['words'].std(), df_incorr['words'].std()], fmt='none', ecolor='k', capsize=10, linewidth=2) 

    ax.set_ylabel('Frecuencia', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    means = [df_corr['words'].mean(), df_incorr['words'].mean()]
    errors = [df_corr['words'].std(), df_incorr['words'].std()]
    x1, x2 = 0, 1

    stat, p_value = mannwhitneyu(df_corr['words'], df_incorr['words'], alternative='two-sided')
    alpha = 0.05  
    if p_value < alpha:
        y = max(means[0] + errors[0], means[1] + errors[1]) +1
        altura_pata = 0.5

        ax.plot([x1, x2], [y, y], lw=1.5, c='k')
        ax.plot([x1, x1], [y - altura_pata, y], lw=1.5, c='k')
        ax.plot([x2, x2], [y - altura_pata, y], lw=1.5, c='k')
        ax.text((x1 + x2) * 0.5, y, "*", ha='center', va='bottom', color='k', fontsize=12)

    fig.savefig(result_path+'barras_dificultad_2.png')
    plt.close('all')

def g_correlacion_tiempo_trial(df_corr, df_incorr):
      
    colors = ['#00FF7D','#FF0700']
    x_T = ['Correctas', 'Incorrectas']
    
    x = np.linspace(0, 50, 50)
    
    r_corr, p_corr = pearsonr( df_corr['time_trial'],  df_corr['long_question'])
    r_corr, p_corr = pearsonr( df_incorr['time_trial'],  df_incorr['long_question'])
    print(r_corr)
    print(p_corr)

    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    fig.suptitle(f'Relación entre el largo de la pregunta y el tiempo de reacción\nTodos los sujetos',  fontsize=16)

    #m, b = np.polyfit(df_corr['time_trial'].tolist(), df_corr['long_question'].tolist(), 1)
    #ax.plot(x, m*x + b, color='#00FF7D', linewidth=2)
    ax.scatter(df_corr['time_trial'].tolist(), df_corr['long_question'].tolist(), color=colors[0], edgecolor='black', linewidth=0.5, zorder=3, label='Correctas')

    #m, b = np.polyfit(df_incorr['time_trial'].tolist(), df_incorr['long_question'].tolist(), 1)
    #ax.plot(x, m*x + b, color='#FF0700', linewidth=2)
    ax.scatter(df_incorr['time_trial'].tolist(), df_incorr['long_question'].tolist(), color=colors[1], edgecolor='black', linewidth=0.5, zorder=3, label='Incorrectas')

    ax.set_ylabel('Largo de la pregunta', fontsize=14)
    ax.set_xlabel('Tiempo [s]', fontsize=14)
    ax.set_xlim(0, 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)

    fig.savefig(result_path+'correlacion_tiempo_trial_1.png')

    #---------------------------
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    fig.suptitle(f'Relación entre cantidad de palabras y el tiempo de reacción\nTodos los sujetos',  fontsize=16)


    #m, b = np.polyfit(df_corr['time_trial'].tolist(), df_corr['words'].tolist(), 1)
    #ax.plot(x, m*x + b, color='#00FF7D', linewidth=2, label='rr')
    ax.scatter(df_corr['time_trial'].tolist(), df_corr['words'].tolist(), color=colors[0], edgecolor='black', linewidth=0.5, zorder=3, label='Correctas')

    #m, b = np.polyfit(df_incorr['time_trial'].tolist(), df_incorr['words'].tolist(), 1)
    #ax.plot(x, m*x + b, color='#FF0700', linewidth=2, label='rr')
    ax.scatter(df_incorr['time_trial'].tolist(), df_incorr['words'].tolist(), color=colors[1], edgecolor='black', linewidth=0.5, zorder=3, label='Inorrectas')

    ax.set_ylabel('Palabras en la pregunta', fontsize=14)
    ax.set_xlabel('Tiempo [s]', fontsize=14)
    ax.set_xlim(0, 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)

    fig.savefig(result_path+'correlacion_tiempo_trial_2.png')
    plt.close('all')

def tabla_tiempos(df_corr, df_incorr):
    
    df_tot = pd.concat([df_corr, df_incorr], ignore_index=True)
    csv_name = 'tabla_tiempos.csv'


    with open(csv_name, 'w', newline='', encoding='utf-8') as archivo_csv:

        escritor_csv = csv.writer(archivo_csv)
        encabezado = ['Sujeto','10%','50% ','90%',
                      '10% tipo:1','50% tipo:1','90% tipo:1',
                      '10% tipo:0','50% tipo:0','90% tipo:0',
                      '10% tipo:3','50% tipo:3','90% tipo:3',
                      '10% tipo:2','50% tipo:2','90% tipo:2']
        escritor_csv.writerow(encabezado)

        for i in df_tot['suj_id'].unique():
            fila_para_csv = [i]
            aux_df  = df_tot[ (df_tot['suj_id'] == i) ]
            times_q = aux_df['time_trial'].tolist() 

            percentiles_a_calcular = [10, 50, 90]
            fila_para_csv.extend(np.percentile(times_q, percentiles_a_calcular))

            for j in df_tot['type_of_question'].unique():
                aux_df  = df_tot[ (df_tot['suj_id'] == i) & (df_tot['type_of_question'] == j) ]
                times_q = aux_df['time_trial'].tolist() 

                fila_para_csv.extend(np.percentile(times_q, percentiles_a_calcular))

            #print(fila_para_csv)
            escritor_csv.writerow(fila_para_csv)

def imagenes_yarbus(x, y, img_name, s):
    # conteo_de_imagenes = df_total['img_name'].value_counts()
    
    # conteo_de_imagenes = conteo_de_imagenes.sort_index()

    # conteo_de_imagenes = conteo_de_imagenes[conteo_de_imagenes == 12]
    # print(conteo_de_imagenes) 
    # imagen seleccionada 0031_00

    img_original = Image.open(os.path.join(img_path, img_name))

    fig, ax = plt.subplots(figsize=(9.6, 5.4), layout='constrained')
    ax.imshow(np.array(img_original))
        
    ax.plot(x, y, color='k', linewidth=1)
    ax.axis('off')

    ax.set_xlim(0,1920)
    ax.set_ylim(1080,0)

    #plt.show()
    plt.savefig(s+'_'+img_name)
    plt.close(fig)

#######################################################################################
################################### Bucle principal ###################################
#######################################################################################

lista_de_carpetas = [nombre for nombre in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, nombre))]
print(f"Las carpetas dentro de '{data_path}' son:")
print(lista_de_carpetas)

df_corr_anw = pd.read_csv('/home/samuel/Documentos/Visual_Reasoning/correct_answers.csv')

cont = 0
df_corr, df_incorr = [], []
for fname in tqdm(lista_de_carpetas, desc="Procesando carpetas"):
    # if cont == 2:
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

    x_tot_0, y_tot_0 = [], []
    x_tot_1, y_tot_1 = [], []
    x_tot_2, y_tot_2 = [], []
    x_tot_3, y_tot_3 = [], []

    #for index, row in df_answ.iterrows():        
        
        # if '0048_00' in row['img_name']:
        #     #print(row['img_name'])
            
        #     t_inicio, t_fin = datos["events"][1][index]/1000+0.2, datos["events"][2][index]/1000
        #     mask_stim = (datos["time_array"] >= t_inicio) & (datos["time_array"] <= t_fin)

        #     # fig, ax = plt.subplots(figsize=(9.6, 5.4), layout='constrained')
        #     # ax.plot(datos["time_array"][mask_stim], datos["pupil_left"][mask_stim])
        #     # plt.show()  
            
        #     #imagenes_yarbus(datos["x_left"][mask_stim], datos["y_left"][mask_stim], row['img_name'], str(index))
        #     #print(df_answ["type_of_question"][index])

        #     if df_answ["type_of_question"][index] == 0:
        #         x_tot_0 = np.concat((x_tot_0, datos["x_left"][mask_stim]))
        #         y_tot_0 = np.concat((y_tot_0, datos["y_left"][mask_stim]))

        #     if df_answ["type_of_question"][index] == 1:
        #         x_tot_1 = np.concat((x_tot_1, datos["x_left"][mask_stim]))
        #         y_tot_1 = np.concat((y_tot_1, datos["y_left"][mask_stim]))

        #     if df_answ["type_of_question"][index] == 2:
        #         x_tot_2 = np.concat((x_tot_2, datos["x_left"][mask_stim]))
        #         y_tot_2 = np.concat((y_tot_2, datos["y_left"][mask_stim]))

        #     if df_answ["type_of_question"][index] == 3:
        #         x_tot_3 = np.concat((x_tot_3, datos["x_left"][mask_stim]))
        #         y_tot_3 = np.concat((y_tot_3, datos["y_left"][mask_stim]))

        #     aux = row['img_name']

    # print('ujui')
    # imagenes_yarbus(x_tot_0, y_tot_0, '0048_000.png', '000')
    # imagenes_yarbus(x_tot_1, y_tot_1, '0048_001.png', '000')
    # imagenes_yarbus(x_tot_2, y_tot_2, '0048_002.png', '000')
    # imagenes_yarbus(x_tot_3, y_tot_3, '0048_003.png', '000')
        
    #break
    df_correctos = df_answ[df_answ['correct'] == 1].copy()
    df_incorrectos = df_answ[df_answ['correct'] == 0].copy()

    df_correctos['suj_id'] = cont
    df_incorrectos['suj_id'] = cont

    df_corr.append(df_correctos)
    df_incorr.append(df_incorrectos)

    cont = cont+1

df_corr = pd.concat(df_corr, ignore_index=True)
df_incorr = pd.concat(df_incorr, ignore_index=True)
df_total = pd.concat([df_corr, df_incorr], ignore_index=True)

#### generación de los gráficos
g_barras_totales(df_corr, df_incorr)
g_barras_totales_grupos(df_corr, df_incorr, lista_de_carpetas)
g_barras_tipos_grupos(df_corr, df_incorr)
g_hist_tiempos_grupos(df_corr, df_incorr)

g_barras_tiempos(df_corr, df_incorr)
g_barras_dificultad(df_corr, df_incorr)
g_correlacion_tiempo_trial(df_corr, df_incorr)

tabla_tiempos(df_corr, df_incorr)


######## 2025-10-14

#imagenes_yarbus(df_total) #0031

################ test estadístico
# # test U de Mann-Whitney
# stat, p_value = mannwhitneyu(df_corr['words'].tolist(), df_incorr['words'].tolist(), alternative='two-sided')

# print(f"Estadístico U: {stat}")
# print(f"P-valor: {p_value}")

# alpha = 0.05  

# print("\n--- Conclusión ---")
# if p_value < alpha:
#     print("Se rechaza la hipótesis nula.")
#     print("Hay una diferencia estadísticamente significativa entre los dos grupos.")
# else:
#     print("No se puede rechazar la hipótesis nula.")
#     print("No hay evidencia suficiente para afirmar que hay una diferencia entre los grupos.")





print("Proceso completado.")

