# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 08:30:39 2024

@author: JMCasado; NBertaina
"""

#General import
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

sys.path.append("../pybrl")

import pybrl as brl

from pydub import AudioSegment

def wav_to_mp3(wav_path, mp3_path):
    sound_mp3 = AudioSegment.from_mp3(wav_path)
    sound_mp3.export(mp3_path, format='wav')

# Local imports
from data_transform import smooth
from data_export.data_export import DataExport
from data_import.data_import import DataImport
from sound_module.simple_sound import simpleSound
from data_transform.predef_math_functions import PredefMathFunctions

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
                    help="Select file type (csv, txt). Defaults to txt.",
                    choices=['csv', 'txt'])
# Receive the directory path from the arguments
parser.add_argument("-d1", "--directory1", type=str,
                    help="Indicate a directory to process as batch.")
parser.add_argument("-d2", "--directory2", type=str,
                    help="Indicate a directory to process as batch.")
parser.add_argument("-d3", "--directory3", type=str,
                    help="Indicate a directory to process as batch.")
parser.add_argument("-d4", "--directory4", type=str,
                    help="Indicate a directory to process as batch.")
# Indicate to save or not the plot
parser.add_argument("-p", "--save-plot", type=bool,
                    help="Indicate if you want to save the plot (False as default)",
                    choices=[False, True])
parser.add_argument("-n", "--noise_snr", type=float,
                    help="Set the signal-to-noise ratio (SNR) for Gaussian noise addition. Defaults to 10.",
                    default=10)

# Alocate the arguments in variables, if extension is empty, select txt as
# default
args = parser.parse_args()
ext = args.file_type or 'txt'
#path = args.directory
path1 = args.directory1
path2 = args.directory2
path3 = args.directory3
path4 = args.directory4
plot_flag = args.save_plot or True
noise_snr = args.noise_snr

# Print a messege if path is not indicated by the user
if not path1:
    print('1At least on intput must be stated.\nUse -h if you need help.')
    exit()
if not path2:
    print('2At least on intput must be stated.\nUse -h if you need help.')
    exit()
if not path3:
    print('3At least on intput must be stated.\nUse -h if you need help.')
    exit()
if not path4:
    print('4At least on intput must be stated.\nUse -h if you need help.')
    exit()
# Format the extension to use it with glob
extension = '*.' + ext

def generate_plot_space(brailleweight=500):
    # Plot without data (cuad1)
    # Generate the blank plot
    figblank = plt.figure()
    axblank = plt.axes()
    axblank.set_title(' ')
    x = brl.translate('x')
    x = brl.toUnicodeSymbols(x, flatten=True)
    axblank.set_xlabel(' ', fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=15)
    y = brl.translate('y')
    y = brl.toUnicodeSymbols(y, flatten=True)
    axblank.set_ylabel(' ', fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=10, rotation=0)
    # Setting ticks
    num0 = brl.translate('0')
    num0 = brl.toUnicodeSymbols(num0, flatten=True)
    num25 = brl.translate('25')
    num25 = brl.toUnicodeSymbols(num25, flatten=True)
    num50 = brl.translate('50')
    num50 = brl.toUnicodeSymbols(num50, flatten=True)
    axblank.set_xticks([0,25,50], 
                        [' ',' ',' '], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight,
                        position=(0,-0.04))
    axblank.set_yticks([0,25,50], 
                        [' ',' ',' '], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight)
    # Resize
    figblank.tight_layout()
    # Save braille figure
    blankplot_path = path[:-4] + 'plot-blank1.png'
    figblank.savefig(blankplot_path)
    plt.close()

    # Plot without data (cuad all)
    # Generate the blank plot
    figblank_all = plt.figure()
    axblank_all = plt.axes()
    axblank_all.set_title(' ')
    x = brl.translate('x')
    x = brl.toUnicodeSymbols(x, flatten=True)
    axblank_all.set_xlabel(x, fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=15)
    y = brl.translate('y')
    y = brl.toUnicodeSymbols(y, flatten=True)
    axblank_all.set_ylabel(y, fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=10, rotation=0)
    # Setting ticks
    num_50 = brl.translate('50')
    caract_resta = [['001001']]
    for i in num_50[0]:
        caract_resta[0].append(i)
    num_50 = caract_resta
    num_50 = brl.toUnicodeSymbols(num_50, flatten=True)
    num0 = brl.translate('0')
    num0 = brl.toUnicodeSymbols(num0, flatten=True)
    num50 = brl.translate('50')
    num50 = brl.toUnicodeSymbols(num50, flatten=True)
    axblank_all.set_xticks([-50,0,50], 
                        [num_50,num0,num50], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight,
                        position=(0,-0.04))
    axblank_all.set_yticks([-50,0,50], 
                        [num_50,num0,num50], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight)
    # Setting limits
    axblank_all.set_xlim(-55,55)
    axblank_all.set_ylim(-55,55)
    # Axis
    axblank_all.axvline(x=0, color='k', linewidth=1)
    axblank_all.axhline(y=0, color='k', linewidth=1)
    # Legend I
    mayus = [['000101']]
    legend1 = brl.translate('i')
    for i in legend1[0]:
        mayus[0].append(i)
    legend1 = brl.toUnicodeSymbols(mayus, flatten=True)
    axblank_all.text(15, 20, legend1, size=24, fontfamily='serif', fontweight=brailleweight, va="bottom", ha="left", rotation=0)
    #Legend II
    mayus = [['000101']]
    legend2 = brl.translate('ii')
    for i in legend2[0]:
        mayus[0].append(i)
    legend2 = brl.toUnicodeSymbols(mayus, flatten=True)
    axblank_all.text(-35, 20, legend2, size=24, fontfamily='serif', fontweight=brailleweight, va="bottom", ha="left", rotation=0)
    #Legend III
    mayus = [['000101']]
    legend3 = brl.translate('iii')
    for i in legend3[0]:
        mayus[0].append(i)
    legend3 = brl.toUnicodeSymbols(mayus, flatten=True)
    axblank_all.text(-40, -30, legend3, size=24, fontfamily='serif', fontweight=brailleweight, va="bottom", ha="left", rotation=0)
    #Legend II
    mayus = [['000101']]
    legend4 = brl.translate('iv')
    for i in legend4[0]:
        mayus[0].append(i)
    legend4 = brl.toUnicodeSymbols(mayus, flatten=True)
    axblank_all.text(15, -30, legend4, size=24, fontfamily='serif', fontweight=brailleweight, va="bottom", ha="left", rotation=0)
    # Resize
    figblank_all.tight_layout()
    # Save braille figure
    blankplot_path = path[:-4] + 'plot-blank-all.png'
    figblank_all.savefig(blankplot_path)
    plt.close()

# Check and display the type of the variable
def check_and_display_type(variable):
    if isinstance(variable, list):
        print(f"The variable is a Python list")
    elif isinstance(variable, np.ndarray):
        print(f"The variable is a NumPy array")
    elif isinstance(variable, pd.Series):
        print(f"The variable is a Pandas Series")
    elif isinstance(variable,pd.DataFrame):
        print(f"The variable is a Pandas DataFrame")
    else:
        print("The variable is not a recognized type.")

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

def generate_braille_plot(data1, data2, data3, data4, plotbraille_path='plot-braille.png', brailleweight=500):
    # Generate the braille plot
    figbraille = plt.figure()

    #First plot
    ax1 = plt.subplot(311)      #ax = plt.subplot(111) para un solo plot
    ax1.plot(data1.loc[:,1], data1.loc[:,2], '#2874a6', linewidth=3) #O5
    #ax1.plot(data_float4.loc[:,1], data_float4.loc[:,2], label='Unknown')
    #plt.tick_params('x', labelsize=6)
    ax1.tick_params('x', labelbottom=False)

    # Second plot
    ax2 = plt.subplot(312, sharex=ax1)
    ax2.plot(data2.loc[:,1], data2.loc[:,2], '#2874a6', linewidth=3) #A5
    #ax2.plot(data_float4.loc[:,1], data_float4.loc[:,2], label='Unknown')
    # make these tick labels invisible
    ax2.tick_params('x', labelbottom=False)

    # Third plot
    ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
    ax3.plot(data3.loc[:,1], data3.loc[:,2], '#2874a6', linewidth=3) #G0
    #ax3.plot(data_float4.loc[:,1], data_float4.loc[:,2], label='Unknown')

    #axbraille = plt.axes()
    # 3 valores de eje x en braille
    abs_val_array = np.abs(data1.loc[:,1] - data1.loc[:,1].min())
    x_pos_min = abs_val_array.idxmin()
    middle = ((data1.loc[:,1].max() - data1.loc[:,1].min())/2) + data1.loc[:,1].min()
    abs_val_array = np.abs(data1.loc[:,1] - middle)
    x_pos_middle = abs_val_array.idxmin()
    abs_val_array = np.abs(data1.loc[:,1] - data1.loc[:,1].max())
    x_pos_max = abs_val_array.idxmin()

    # primer numero del eje x
    xinicio_text = numinbraille(data1.loc[x_pos_min,1])
    # numero medio del eje x
    xmedio_text = numinbraille(data1.loc[x_pos_middle,1])
    # numero final del eje x
    xfinal_text = numinbraille(data1.loc[x_pos_max,1])
    
    ax3.set_xticks([data1.loc[x_pos_min,1],data1.loc[x_pos_middle,1],data1.loc[x_pos_max,1]], 
                        [xinicio_text,xmedio_text,xfinal_text], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight,
                        position=(0,-0.04))

    # 3 valores de eje y en braille
    # Found min, middle, max possitions and values
    abs_val_array = np.abs(data1.loc[:,2] - data1.loc[:,2].min())
    y_pos_min = abs_val_array.idxmin()
    middle = ((data1.loc[:,2].max() - data1.loc[:,2].min())/2) + data1.loc[:,2].min()
    abs_val_array = np.abs(data1.loc[:,2] - middle)
    y_pos_middle = abs_val_array.idxmin()
    abs_val_array = np.abs(data1.loc[:,2] - data1.loc[:,2].max())
    y_pos_max = abs_val_array.idxmin()

    y_pos_min_text = numinbraille(data1.loc[y_pos_min,2])
    y_pos_middle_text = numinbraille(data1.loc[y_pos_middle,2])
    y_pos_max_text = numinbraille(data1.loc[y_pos_max,2])
    ax1.set_yticks([data1.loc[y_pos_min,2],data1.loc[y_pos_max,2]], 
                        [y_pos_min_text,y_pos_max_text], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight)
    ax2.set_yticks([data1.loc[y_pos_min,2],data1.loc[y_pos_max,2]], 
                        [y_pos_min_text,y_pos_max_text], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight)
    ax3.set_yticks([data1.loc[y_pos_min,2],data1.loc[y_pos_max,2]], 
                        [y_pos_min_text,y_pos_max_text], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight)

    ax1.set_title(' ')
    x = brl.translate('x')
    x = brl.toUnicodeSymbols(x, flatten=True)
    ax3.set_xlabel(x, fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=15)
    y = brl.translate('y')
    y = brl.toUnicodeSymbols(y, flatten=True)
    ax2.set_ylabel(y, fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=10, rotation=0)

    


    #axbraille.plot(dataframe.loc[:, 0], dataframe.loc[:, 1], '#2874a6', linewidth=3)
    # Ejes de coordenadas
    #if dataframe.loc[:, 0].min() < 0 and dataframe.loc[:, 0].max() > 0:
    #    axbraille.axvline(x=0, color='k', linewidth=1)
    #if dataframe.loc[:, 1].min() < 0 and dataframe.loc[:, 1].max() > 0:
    #    axbraille.axhline(y=0, color='k', linewidth=1)
    # Resize
    figbraille.tight_layout()
    # Save braille figure
    figbraille.savefig(plotbraille_path)
    plt.close()

# Create an empty figure or plot to save it
fig = plt.figure()
# Defining the axes so that we can plot data into it.
#ax = plt.axes()

# Open each file
#data, status, msg = _dataimport.set_arrayfromfile(path, ext)
data1, status, msg = _dataimport.set_arrayfromfile(path1, ext)
data2, status, msg = _dataimport.set_arrayfromfile(path2, ext)
data3, status, msg = _dataimport.set_arrayfromfile(path3, ext)
data4, status, msg = _dataimport.set_arrayfromfile(path4, ext)
# Check if the import is correct
if data1.shape[1]<2:
    print("Error reading file 1, only detect one column.")
    exit()
if data2.shape[1]<2:
    print("Error reading file 2, only detect one column.")
    exit()
if data3.shape[1]<2:
    print("Error reading file 3, only detect one column.")
    exit()
if data4.shape[1]<2:
    print("Error reading file 4, only detect one column.")
    exit()
# Extract the names and turn to float
data_float1 = data1.iloc[1:, 1:].astype(float)
data_float2 = data2.iloc[1:, 1:].astype(float)
data_float3 = data3.iloc[1:, 1:].astype(float)
data_float4 = data4.iloc[1:, 1:].astype(float)

#Inicializamos la posición para no tener error cuando no haya recortes
x_pos_min = 0

# Cut first data set
abs_val_array = np.abs(data_float1.loc[:,1] - 3700)
x_pos_min = abs_val_array.idxmin()
abs_val_array = np.abs(data_float1.loc[:,1] - 4700)
x_pos_max = abs_val_array.idxmin()
data_float1 = data1.iloc[x_pos_min:x_pos_max, :].astype(float)
# Cut second data set
data_float2 = data2.iloc[x_pos_min:x_pos_max, :].astype(float)
# Cut third data set
data_float3 = data3.iloc[x_pos_min:x_pos_max, :].astype(float)
# Cut fourth data set
data_float4 = data4.iloc[x_pos_min:x_pos_max, :].astype(float)

# Para presentar los datos en nm en lugar de Armstrong
data_float1.loc[:,1] = data_float1.loc[:,1] / 10
data_float2.loc[:,1] = data_float2.loc[:,1] / 10
data_float3.loc[:,1] = data_float3.loc[:,1] / 10
data_float4.loc[:,1] = data_float4.loc[:,1] / 10

#First plot
ax1 = plt.subplot(311)      #ax = plt.subplot(111) para un solo plot
ax1.plot(data_float1.loc[:,1], data_float1.loc[:,2], label='O5 V')
#ax1.plot(data_float4.loc[:,1], data_float4.loc[:,2], label='Unknown')
#plt.tick_params('x', labelsize=6)
ax1.tick_params('x', labelbottom=False)

# Second plot
ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(data_float2.loc[:,1], data_float2.loc[:,2], label='A5 V')
#ax2.plot(data_float4.loc[:,1], data_float4.loc[:,2], label='Unknown')
# make these tick labels invisible
ax2.tick_params('x', labelbottom=False)

# Third plot
ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
ax3.plot(data_float3.loc[:,1], data_float3.loc[:,2], label='G0 V')
#ax3.plot(data_float4.loc[:,1], data_float4.loc[:,2], label='Unknown')

ax1.legend()
ax2.legend()
ax3.legend()

plot_path = path1[:-6] + 'plot.png'
fig.savefig(plot_path)

# Generate de plot
#ax.set_xlabel('x')
#ax.set_ylabel('y', rotation=0)
# Separate the name file from the path to set the plot title
#filename = os.path.basename(path)
# Plot 
#ax.plot(data_float.loc[:, 0], data_float.loc[:, 1], '#2874a6', linewidth=3)
# Ejes de coordenadas
#if data_float.loc[:, 0].min() < 0:
#    ax.axvline(x=0, color='k', linewidth=1)
#if data_float.loc[:, 1].min() < 0:
#    ax.axhline(y=0, color='k', linewidth=1)
# Set the path to save the plot and save it
#plot_path = path[:-4] + 'plot.png'
#fig.savefig(plot_path)
plt.close()

plotbraille_path = path1[:-6] + 'plotbraille.png'
generate_braille_plot(data_float1, data_float2, data_float3, data_float4, plotbraille_path)

# Reproduction
# Normalize the data to sonify
x1, y1, status = _math.normalize(data_float1.loc[:,1], data_float1.loc[:,2], init=x_pos_min)
x2, y2, status = _math.normalize(data_float2.loc[:,1], data_float2.loc[:,2], init=x_pos_min)
x3, y3, status = _math.normalize(data_float3.loc[:,1], data_float3.loc[:,2], init=x_pos_min)
x4, y4, status = _math.normalize(data_float4.loc[:,1], data_float4.loc[:,2], init=x_pos_min)

# Save sound
wav_name1 = path1[:-6] + 'O5.wav'
wav_name2 = path1[:-6] + 'A5.wav'
wav_name3 = path1[:-6] + 'G0.wav'
wav_name4 = path1[:-6] + 'unknown.wav'
mp3_name1 = path1[:-6] + 'O5.mp3'
mp3_name2 = path1[:-6] + 'A5.mp3'
mp3_name3 = path1[:-6] + 'G0.mp3'
mp3_name4 = path1[:-6] + 'unknown.mp3'
_simplesound.save_sound(wav_name1, data_float1.loc[:,1], y1, init=x_pos_min)
wav_to_mp3(wav_name1, mp3_name1)
_simplesound.save_sound(wav_name2, data_float2.loc[:,1], y2, init=x_pos_min)
wav_to_mp3(wav_name2, mp3_name2)
_simplesound.save_sound(wav_name3, data_float3.loc[:,1], y3, init=x_pos_min)
wav_to_mp3(wav_name3, mp3_name3)
_simplesound.save_sound(wav_name4, data_float4.loc[:,1], y4, init=x_pos_min)
wav_to_mp3(wav_name4, mp3_name4)
# Para sonidos combinados las siguientes líneas
#_simplesound.save_sound_multicol_stars(wav_name1, data_float1.loc[:,1], y1, y4, init=x_pos_min) 
#_simplesound.save_sound_multicol_stars(wav_name2, data_float1.loc[:,1], y2, y4, init=x_pos_min)
#_simplesound.save_sound_multicol_stars(wav_name3, data_float1.loc[:,1], y3, y4, init=x_pos_min)
# Print time
now = datetime.datetime.now()
print(now.strftime('%Y-%m-%d_%H-%M-%S'))
