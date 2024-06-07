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
_simplesound.reproductor.set_time_base(0.1)
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
# Alocate the arguments in variables, if extension is empty, select txt as
# default
args = parser.parse_args()
ext = args.file_type or 'txt'
path = args.directory
plot_flag = args.save_plot or False
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
    if (floatnum < 0) and (int(abs(floatnum)) == 0):
        num = str(1)
    else:
        num = str(int(abs(floatnum)))
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

def generate_braille_plot(x, y, name='plot-braille.png', brailleweight=500):
    # Generate the braille plot
    figbraille = plt.figure()
    axbraille = plt.axes()
    # 3 valores de eje x en braille
    abs_val_array = np.abs(x.iloc[:,0] - x.iloc[:,0].min())
    x_pos_min = abs_val_array.idxmin()
    middle = ((x.iloc[:,0].max() - x.iloc[:,0].min())/2) + x.iloc[:,0].min()
    abs_val_array = np.abs(x.iloc[:,0] - middle)
    x_pos_middle = abs_val_array.idxmin()
    abs_val_array = np.abs(x.iloc[:,0] - x.iloc[:,0].max())
    x_pos_max = abs_val_array.idxmin()

    # primer numero del eje x
    xinicio_text = numinbraille(x.iloc[x_pos_min,0])
    # numero medio del eje x
    xmedio_text = numinbraille(x.iloc[x_pos_middle,0])
    # numero final del eje x
    xfinal_text = numinbraille(x.iloc[x_pos_max,0])
    
    #axbraille.set_xticks([dataframe.iloc[x_pos_min,0],dataframe.iloc[x_pos_middle,0],dataframe.iloc[x_pos_max,0]], 
    #                    [xinicio_text,xmedio_text,xfinal_text], 
    #                    fontsize=24,
    #                    fontfamily='serif',
    #                    fontweight=brailleweight,
    #                    position=(0,-0.04))

    # 3 valores de eje y en braille
    # Found min, middle, max possitions and values
    abs_val_array = np.abs(y.iloc[:,1] - y.iloc[:,1].min())
    y_pos_min = abs_val_array.idxmin()
    middle = ((y.iloc[:,1].max() - y.iloc[:,1].min())/2) + y.iloc[:,1].min()
    abs_val_array = np.abs(y.iloc[:,1] - middle)
    y_pos_middle = abs_val_array.idxmin()
    abs_val_array = np.abs(y.iloc[:,1] - y.iloc[:,1].max())
    y_pos_max = abs_val_array.idxmin()

    y_pos_min_text = numinbraille(y.iloc[y_pos_min,1])
    y_pos_middle_text = numinbraille(y.iloc[y_pos_middle,1])
    y_pos_max_text = numinbraille(y.iloc[y_pos_max,1])
    #axbraille.set_yticks([dataframe.iloc[y_pos_min,1],dataframe.iloc[y_pos_middle,1],dataframe.iloc[y_pos_max,1]], 
    #                    [y_pos_min_text,y_pos_middle_text,y_pos_max_text], 
    #                    fontsize=24,
    #                    fontfamily='serif',
    #                    fontweight=brailleweight)

    axbraille.set_title(' ')
    x = brl.translate('x')
    x = brl.toUnicodeSymbols(x, flatten=True)
    axbraille.set_xlabel(x, fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=15)
    y = brl.translate('y')
    y = brl.toUnicodeSymbols(y, flatten=True)
    axbraille.set_ylabel(y, fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=10, rotation=0)
    axbraille.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1])#, '#2874a6', linewidth=3)
    # Ejes de coordenadas
    if dataframe.iloc[:, 0].min() < 0 and dataframe.iloc[:, 0].max() > 0:
        axbraille.axvline(x=0, color='k', linewidth=1)
    if dataframe.iloc[:, 1].min() < 0 and dataframe.iloc[:, 1].max() > 0:
        axbraille.axhline(y=0, color='k', linewidth=1)
    # Resize
    figbraille.tight_layout()
    # Save braille figure
    brailleplot_path = path[:-4] + name
    figbraille.savefig(brailleplot_path)
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
    periodo_CGCas = 4.3652815
    periodo_RWPhe = 5.4134367
    t0_CGCas = 2457412.70647
    t0_RWPhe = 2458053.49761

    # """Para CGCas"""
    # new_df.loc[:,2] = (new_df.loc[:,2].astype(float) - t0_CGCas) / periodo_CGCas
    # new_df.loc[:,2] = (new_df.loc[:,2] - new_df.loc[:,2].astype(float).astype(int)) + 0.79
    # for i in range (1,(len(new_df.loc[:,2]))):
    #     if new_df.loc[i,2] < 0:
    #         new_df.loc[i,2] = new_df.loc[i,2] + 2
    """Para RWPhe"""
    new_df.loc[:,0] = (new_df.loc[:,0].astype(float) - t0_RWPhe) / periodo_RWPhe
    new_df.loc[:,0] = (new_df.loc[:,0] - new_df.loc[:,0].astype(float).astype(int)) + 0.55
    for i in range (1,(len(new_df.loc[:,0]))):
        if new_df.loc[i,0] < 0:
            new_df.loc[i,0] = new_df.loc[i,0] + 2
    
    
    new_df.loc[:,2] = new_df.loc[:,2].astype(float)
    
    sort_df = pd.DataFrame(new_df).sort_values(0, axis=0)
        
    yl = sort_df.loc[:,2].values
    yhat = smooth.savitzky_golay(yl, 51, 7)
    
    
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
        #ax.set_title('CG-Cas-Cepheid')
        ax.set_title('RW-Phe-Eclipsing Binary')
        # xnumpy = xnumpy / 10
        # ax.scatter(xnumpy, ynumpy)
        # ax.plot(sort_df.loc[:,2], sort_df.loc[:,4], 'o')
        ax.scatter(sort_df.loc[:,0], yhat)        
        # Set the path to save the plot and save it
        plot_path = path + '/' + os.path.basename(filename) + '_plot.png'
        fig.savefig(plot_path)

        # Generate the dataFrame to plot with braille
        data_float = sort_df.loc[:, 0].to_frame()
        df = pd.DataFrame(yhat)
        data_float = data_float.join(df, rsuffix='1')
        generate_braille_plot(sort_df.loc[:,0], yhat, 'plot-braille1.png')
        
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
    _simplesound.save_sound(wav_name, x, y)
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d_%H-%M-%S'))
    i = i + 1

