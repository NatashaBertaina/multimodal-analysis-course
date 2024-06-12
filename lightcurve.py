# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:13:42 2022

@author: johi-
"""

import os
import sys
import argparse
import glob
import numpy as np
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import pandas as pd
from data_transform import smooth
from data_export.data_export import DataExport
from data_import.data_import import DataImport
from sound_module.simple_sound import simpleSound
from data_transform.predef_math_functions import PredefMathFunctions

sys.path.append("../pybrl")

import pybrl as brl

from pydub import AudioSegment

def wav_to_mp3(wav_path, mp3_path):
    sound_mp3 = AudioSegment.from_mp3(wav_path)
    sound_mp3.export(mp3_path, format='wav')

# Instanciate the sonoUno clases needed
_dataexport = DataExport(False)
_dataimport = DataImport()
_simplesound = simpleSound()
_math = PredefMathFunctions()
# Sound configurations, predefined at the moment
_simplesound.reproductor.set_continuous()
_simplesound.reproductor.set_waveform('sine')  # piano; sine
_simplesound.reproductor.set_time_base(0.03)
_simplesound.reproductor.set_min_freq(380)
_simplesound.reproductor.set_max_freq(800)
# The argparse library is used to pass the path and extension where the data
# files are located
parser = argparse.ArgumentParser()
# Receive the extension from the arguments
parser.add_argument("-t", "--file-type", type=str,
                    help="Select file type.",
                    choices=['csv', 'txt'])
# Receive the directory path from the arguments
parser.add_argument("-d", "--directory", type=str,
                    help="Indicate a directory to process as batch.")
# Indicate to save or not the plot
parser.add_argument("-p", "--save-plot", type=bool,
                    help="Indicate if you want to save the plot (False as default)",
                    choices=[False,True])
# Indicate the star type
parser.add_argument("-s", "--star-type", type=str,
                    help="Indicate the star type to plot (RWPhe, V0748Cep, ZLep, CGCas, HWPup, MNCam)",
                    choices=['RWPhe', 'V0748Cep', 'ZLep', 'CGCas', 'HWPup', 'MNCam'])
# Alocate the arguments in variables, if extension is empty, select txt as
# default
args = parser.parse_args()
ext = args.file_type or 'csv'
path = args.directory
plot_flag = args.save_plot or True
starType = args.star_type
# Print a messege if path is not indicated by the user
if not path:
    print('At least on intput must be stated.\nUse -h if you need help.')
    exit()
# Format the extension to use it with glob
extension = '*.' + ext
# Initialize a counter to show a message during each loop
i = 1
if plot_flag:
    # Create an empty figure or plot to save it
    cm = 1/2.54  # centimeters in inches
    fig = plt.figure(figsize=(15*cm, 10*cm), dpi=300)
    # Defining the axes so that we can plot data into it.
    ax = plt.axes()
# Loop to walk the directory and sonify each data file
now = datetime.datetime.now()
print(now.strftime('%Y-%m-%d_%H-%M-%S'))


def numinbraille(floatnum):
    num_primera_serie = [['010110'], 
                         ['100000'], 
                         ['110000'],
                         ['100100'],
                         ['100110'],
                         ['100010'],
                         ['110100'],
                         ['110110'],
                         ['110010'],
                         ['010100']]
    simbolo_num = [['001111']]
    simbolo_resta = [['001001']]
    # convertion
    totext = [simbolo_num[0].copy()]
    if (floatnum < 0) and (round(abs(floatnum)) == 0):
        num = str(1)
    else:
        num = str(round(abs(floatnum)))
    for i in num:
        a = num_primera_serie[int(i)]
        totext[0].append(a[0])
    if floatnum < 0:
        totext2 = [simbolo_resta[0].copy()]
        for i in totext[0]:
            totext2[0].append(i)
        totext2 = brl.toUnicodeSymbols(totext2, flatten=True)
        return totext2
    totext = brl.toUnicodeSymbols(totext, flatten=True)
    return totext

def generate_braille_plot(x,y, plot_braille_path='plot-braille.png', brailleweight=500):
    # Generate the braille plot
    figbraille = plt.figure()
    axbraille = plt.axes()
    # 3 valores de eje x en braille
    #abs_val_array = np.abs(dataframe.iloc[:,0] - dataframe.iloc[:,0].min())
    abs_val_array = np.abs(x - x.min())
    x_pos_min = abs_val_array.idxmin()
    #middle = ((dataframe.iloc[:,0].max() - dataframe.iloc[:,0].min())/2) + dataframe.iloc[:,0].min()
    middle = ((x.max() - x.min())/2) + x.min()
    #abs_val_array = np.abs(dataframe.iloc[:,0] - middle)
    abs_val_array = np.abs(x - middle)
    x_pos_middle = abs_val_array.idxmin()
    #abs_val_array = np.abs(dataframe.iloc[:,0] - dataframe.iloc[:,0].max())
    abs_val_array = np.abs(x - x.max())
    x_pos_max = abs_val_array.idxmin()

    # primer numero del eje x
    #xinicio_text = numinbraille(dataframe.iloc[x_pos_min,0])
    xinicio_text = numinbraille(x[x_pos_min])
    # numero medio del eje x
    #xmedio_text = numinbraille(dataframe.iloc[x_pos_middle,0])
    xmedio_text = numinbraille(x[x_pos_middle])
    # numero final del eje x
    #xfinal_text = numinbraille(dataframe.iloc[x_pos_max,0])
    xfinal_text = numinbraille(x[x_pos_max])
    
    axbraille.set_xticks([x[x_pos_min],x[x_pos_middle],x[x_pos_max]], 
                        [xinicio_text,xmedio_text,xfinal_text], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight,
                        position=(0,-0.04))

    # 3 valores de eje y en braille
    # Found min, middle, max possitions and values
    #abs_val_array = np.abs(dataframe.iloc[:,1] - dataframe.iloc[:,1].min())
    abs_val_array = np.abs(y - y.min())
    y_pos_min = abs_val_array.argmin()
    #middle = ((dataframe.iloc[:,1].max() - dataframe.iloc[:,1].min())/2) + dataframe.iloc[:,1].min()
    middle = ((y.max() - y.min())/2) + y.min()
    #abs_val_array = np.abs(dataframe.iloc[:,1] - middle)
    abs_val_array = np.abs(y - middle)
    y_pos_middle = abs_val_array.argmin()
    #abs_val_array = np.abs(dataframe.iloc[:,1] - dataframe.iloc[:,1].max())
    abs_val_array = np.abs(y - y.max())
    y_pos_max = abs_val_array.argmin()

    #y_pos_min_text = numinbraille(dataframe.iloc[y_pos_min,1])
    y_pos_min_text = numinbraille(y[y_pos_min])
    #y_pos_middle_text = numinbraille(dataframe.iloc[y_pos_middle,1])
    y_pos_middle_text = numinbraille(y[y_pos_middle])
    #y_pos_max_text = numinbraille(dataframe.iloc[y_pos_max,1])
    y_pos_max_text = numinbraille(y[y_pos_max])
    axbraille.set_yticks([y[y_pos_min],y[y_pos_max]], 
                        [y_pos_min_text,y_pos_max_text], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight)

    axbraille.set_title(' ')
    #axbraille.set_xlabel('Phase')
    x_label = brl.translate('fase')
    x_label = brl.toUnicodeSymbols(x_label, flatten=True)
    axbraille.set_xlabel(x_label, fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=15)
    #axbraille.set_ylabel('Mag')
    y_label = brl.translate('mag')
    y_label = brl.toUnicodeSymbols(y_label, flatten=True)
    axbraille.set_ylabel(y_label, fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=10)
    #axbraille.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1])#, '#2874a6', linewidth=3)
    axbraille.invert_yaxis()
    axbraille.scatter(x,y)
    # Ejes de coordenadas
    #if dataframe.iloc[:, 0].min() < 0 and dataframe.iloc[:, 0].max() > 0:
    #    axbraille.axvline(x=0, color='k', linewidth=1)
    #if dataframe.iloc[:, 1].min() < 0 and dataframe.iloc[:, 1].max() > 0:
    #    axbraille.axhline(y=0, color='k', linewidth=1)
    # Resize
    figbraille.tight_layout()
    # Save braille figure
    figbraille.savefig(plot_braille_path)
    plt.close()


for filename in glob.glob(os.path.join(path, extension)):
    print("Converting data file number "+str(i)+" to sound.")
    # Open each file
    data, status, msg = _dataimport.set_arrayfromfile(filename, ext)
    # Convert into numpy, split in x and y and normalyze
    if data.shape[1]<2:
        print("Error reading file, only detect one column.")
        exit()
    data = data.iloc[1:, :]
    # x = data.loc[1:, 2]
    # xnumpy = x.values.astype(np.float64)
    # y = data.loc[1:, 4]
    # ynumpy = y.values.astype(np.float64)
    
    #Select columns to order next
    selected_columns = data[[0,2]]
    new_df = selected_columns.copy()
    # sort_df = pd.DataFrame(new_df).sort_values(2, axis=0)
    
    
    # x = sort_df[:,0].astype(np.float64)
    # y = sort_df[:,1].astype(np.float64)
    
    # x = sort_df.loc[1:, 0]
    # xnumpy = x.values.astype(np.float64)
    # y = sort_df.loc[1:, 1]
    # ynumpy = y.values.astype(np.float64)
    
    # Para estrellas variables
    """https://asas-sn.osu.edu/variables/753bdd73-38a7-5e43-b6c0-063292c7f28d"""
    periodo_CGCas = 4.3652815
    t0_CGCas = 2457412.70647
    """https://asas-sn.osu.edu/variables/dfa51488-c6b7-5a03-abd4-df3c28273250"""
    periodo_RWPhe = 5.4134367
    t0_RWPhe = 2458053.49761
    """https://asas-sn.osu.edu/variables/dfcbcf52-8a62-542f-9383-1c712d7c042c"""
    periodo_V0748Cep = 2.5093526
    t0_V0748Cep = 2458024.93242
    """https://asas-sn.osu.edu/variables/70cc7024-5027-52f9-a834-75c51f4a5064"""
    periodo_ZLep = 0.9937068
    t0_ZLep = 2457699.6236
    """https://asas-sn.osu.edu/variables/c3faa9d0-6e10-5775-8bb0-075defcd2578"""
    periodo_MNCam = 8.1796049
    t0_MNCam = 2458046.08639
    """https://asas-sn.osu.edu/variables/2083f661-73f5-512f-aee5-fd7ad26d5b30"""
    periodo_HWPup = 13.4590914
    t0_HWPup = 2457786.63153

    #starType = 'ZLep'
    if starType == 'CGCas':
        new_df.loc[:,0] = (new_df.loc[:,0].astype(float) - t0_CGCas) / periodo_CGCas
        new_df.loc[:,0] = (new_df.loc[:,0] - new_df.loc[:,0].astype(float).astype(int)) + 0.79
    elif starType == 'RWPhe':
        new_df.loc[:,0] = (new_df.loc[:,0].astype(float) - t0_RWPhe) / periodo_RWPhe
        new_df.loc[:,0] = (new_df.loc[:,0] - new_df.loc[:,0].astype(float).astype(int)) + 0.55
    elif starType == 'V0748Cep':
        new_df.loc[:,0] = (new_df.loc[:,0].astype(float) - t0_V0748Cep) / periodo_V0748Cep
        new_df.loc[:,0] = (new_df.loc[:,0] - new_df.loc[:,0].astype(float).astype(int)) + 0.45
    elif starType == 'ZLep':
        new_df.loc[:,0] = (new_df.loc[:,0].astype(float) - t0_ZLep) / periodo_ZLep
        new_df.loc[:,0] = (new_df.loc[:,0] - new_df.loc[:,0].astype(float).astype(int))
    elif starType == 'MNCam':
        new_df.loc[:,0] = (new_df.loc[:,0].astype(float) - t0_MNCam) / periodo_MNCam
        new_df.loc[:,0] = (new_df.loc[:,0] - new_df.loc[:,0].astype(float).astype(int)) + 0.45
    elif starType == 'HWPup':
        new_df.loc[:,0] = (new_df.loc[:,0].astype(float) - t0_HWPup) / periodo_HWPup
        new_df.loc[:,0] = (new_df.loc[:,0] - new_df.loc[:,0].astype(float).astype(int)) + 0.10
    else:
        print('Error en el tipo de estrella.')

    for i in range (1,(len(new_df.loc[:,0])+1)):
        if new_df.loc[i,0] < 0:
            new_df.loc[i,0] = new_df.loc[i,0] + 2
        
    new_df.loc[:,2] = new_df.loc[:,2].astype(float)
    
    sort_df = pd.DataFrame(new_df).sort_values(0, axis=0)
        
    yhat = sort_df.loc[:,2].values
    #yhat = smooth.savitzky_golay(yl, 51, 7)
    
    
    #x, y, status = _math.normalize(sort_df.loc[:,2], sort_df.loc[:,4])
    #x, y, status = _math.normalize(sort_df.loc[:,2], yhat)
    if plot_flag:
        # Configure axis, plot the data and save it
        # Erase the plot
        ax.cla()
        # First file of the column is setted as axis name
        x_name = str(data.iloc[0,0])
        ax.set_xlabel('Phase')
        y_name = str(data.iloc[0,0])
        ax.set_ylabel('Mag')
        ax.invert_yaxis()
        # Separate the name file from the path to set the plot title
        head, tail = os.path.split(filename)
        if starType == 'CGCas':
            ax.set_title('CG-Cas-Cepheid')
        elif starType == 'RWPhe':
            ax.set_title('RW-Phe-Eclipsing Binary')
        elif starType == 'V0748Cep':
            ax.set_title('V0748-Cep-Eclipsing Binary')
        elif starType == 'ZLep':
            ax.set_title('Z-Lep-Eclipsing Binary')
        elif starType == 'MNCam':
            ax.set_title('MN-Cam-Cepheid')
        elif starType == 'HWPup':
            ax.set_title('HW-Pup-Cepheid')
        else:
            ax.set_title(' ')
        # xnumpy = xnumpy / 10
        # ax.scatter(xnumpy, ynumpy)
        # ax.plot(sort_df.loc[:,2], sort_df.loc[:,4], 'o')
        ax.scatter(sort_df.loc[:,0], yhat, marker='.')        
        # Set the path to save the plot and save it
        plot_path = path + '/' + os.path.basename(filename) + '_plot.png'
        fig.savefig(plot_path)

        # Generate the dataFrame to plot with braille
        data_float = sort_df.loc[:, 0].to_frame()
        df = pd.DataFrame(yhat)
        data_float = data_float.join(df, rsuffix='1')
        plot_braille_path = path + '/' + os.path.basename(filename) + '_plot-braille.png'
        generate_braille_plot(sort_df.loc[:,0], yhat, plot_braille_path)
        
        #Tratando de invertir valores
        linea_media = (np.nanmax(yhat) - np.nanmin(yhat))/2 + np.nanmin(yhat)
        
        ax.axhline(y = linea_media, xmin = 0, xmax = 2)
        
        cont = 0
        for i in yhat:
            if i > linea_media:
                yhat[cont] = linea_media - (i - linea_media)
            if i == linea_media:
                yhat[cont] = i
            if i < linea_media:
                yhat[cont] = linea_media + (linea_media - i)
            cont = cont + 1
        
        #ax.scatter(sort_df.loc[:,0], yhat)
        
        #ax.scatter(sort_df.loc[:,2], linea_media)  
        
        # Set the path to save the plot and save it, se comenta porque solo se
        # uso como control
        #plot_path = path + '/' + os.path.basename(filename) + '_plot2.png'
        #fig.savefig(plot_path)
        
    # Save the sound
    
    x, y, status = _math.normalize(sort_df.loc[:,0], yhat)
    
    wav_name = path + '/' + os.path.basename(filename) + '_sound.wav'
    mp3_name = path + '/' + os.path.basename(filename) + '_sound.mp3'
    _simplesound.save_sound(wav_name, x, y)
    wav_to_mp3(wav_name, mp3_name)
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d_%H-%M-%S'))
    i = i + 1

