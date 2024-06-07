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
parser.add_argument("-d", "--directory", type=str,
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
path = args.directory
plot_flag = args.save_plot or True
noise_snr = args.noise_snr

# Print a messege if path is not indicated by the user
if not path:
    print('1At least on intput must be stated.\nUse -h if you need help.')
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

def generate_braille_plot(dataframe, name='plot-braille.png', brailleweight=500):
    # Generate the braille plot
    figbraille = plt.figure()
    axbraille = plt.axes()
    # 3 valores de eje x en braille
    abs_val_array = np.abs(dataframe.loc[:,0] - dataframe.loc[:,0].min())
    x_pos_min = abs_val_array.idxmin()
    middle = ((dataframe.loc[:,0].max() - dataframe.loc[:,0].min())/2) + dataframe.loc[:,0].min()
    abs_val_array = np.abs(dataframe.loc[:,0] - middle)
    x_pos_middle = abs_val_array.idxmin()
    abs_val_array = np.abs(dataframe.loc[:,0] - dataframe.loc[:,0].max())
    x_pos_max = abs_val_array.idxmin()

    # primer numero del eje x
    xinicio_text = numinbraille(dataframe.loc[x_pos_min,0])
    # numero medio del eje x
    xmedio_text = numinbraille(dataframe.loc[x_pos_middle,0])
    # numero final del eje x
    xfinal_text = numinbraille(dataframe.loc[x_pos_max,0])
    
    axbraille.set_xticks([dataframe.loc[x_pos_min,0],dataframe.loc[x_pos_middle,0],dataframe.loc[x_pos_max,0]], 
                        [xinicio_text,xmedio_text,xfinal_text], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight,
                        position=(0,-0.04))

    # 3 valores de eje y en braille
    # Found min, middle, max possitions and values
    abs_val_array = np.abs(dataframe.loc[:,1] - dataframe.loc[:,1].min())
    y_pos_min = abs_val_array.idxmin()
    middle = ((dataframe.loc[:,1].max() - dataframe.loc[:,1].min())/2) + dataframe.loc[:,1].min()
    abs_val_array = np.abs(dataframe.loc[:,1] - middle)
    y_pos_middle = abs_val_array.idxmin()
    abs_val_array = np.abs(dataframe.loc[:,1] - dataframe.loc[:,1].max())
    y_pos_max = abs_val_array.idxmin()

    y_pos_min_text = numinbraille(dataframe.loc[y_pos_min,1])
    y_pos_middle_text = numinbraille(dataframe.loc[y_pos_middle,1])
    y_pos_max_text = numinbraille(dataframe.loc[y_pos_max,1])
    axbraille.set_yticks([dataframe.loc[y_pos_min,1],dataframe.loc[y_pos_middle,1],dataframe.loc[y_pos_max,1]], 
                        [y_pos_min_text,y_pos_middle_text,y_pos_max_text], 
                        fontsize=24,
                        fontfamily='serif',
                        fontweight=brailleweight)

    axbraille.set_title(' ')
    x = brl.translate('x')
    x = brl.toUnicodeSymbols(x, flatten=True)
    axbraille.set_xlabel(x, fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=15)
    y = brl.translate('y')
    y = brl.toUnicodeSymbols(y, flatten=True)
    axbraille.set_ylabel(y, fontsize=24, fontfamily='serif', fontweight=brailleweight, labelpad=10, rotation=0)
    axbraille.plot(dataframe.loc[:, 0], dataframe.loc[:, 1], '#2874a6', linewidth=3)
    # Ejes de coordenadas
    if dataframe.loc[:, 0].min() < 0 and dataframe.loc[:, 0].max() > 0:
        axbraille.axvline(x=0, color='k', linewidth=1)
    if dataframe.loc[:, 1].min() < 0 and dataframe.loc[:, 1].max() > 0:
        axbraille.axhline(y=0, color='k', linewidth=1)
    # Resize
    figbraille.tight_layout()
    # Save braille figure
    brailleplot_path = path[:-4] + name
    figbraille.savefig(brailleplot_path)
    plt.close()

# Create an empty figure or plot to save it
fig = plt.figure()
# Defining the axes so that we can plot data into it.
ax = plt.axes()

# Open each file
data, status, msg = _dataimport.set_arrayfromfile(path, ext)
# Check if the import is correct
if data.shape[1]<2:
    print("Error reading file 1, only detect one column.")
    exit()
# Extract the names and turn to float
data_float = data.iloc[1:, :].astype(float)
x_pos_min = 1

# Generate de plot
ax.set_xlabel('x')
ax.set_ylabel('y', rotation=0)
# Separate the name file from the path to set the plot title
filename = os.path.basename(path)
# Plot 
ax.plot(data_float.loc[:, 0], data_float.loc[:, 1], '#2874a6', linewidth=3)
# Ejes de coordenadas
if data_float.loc[:, 0].min() < 0:
    ax.axvline(x=0, color='k', linewidth=1)
if data_float.loc[:, 1].min() < 0:
    ax.axhline(y=0, color='k', linewidth=1)
# Set the path to save the plot and save it
plot_path = path[:-4] + 'plot.png'
fig.savefig(plot_path)
plt.close()

generate_braille_plot(data_float, 'plot-braille1.png')

# Reproduction
# Normalize the data to sonify
x1, y1, status = _math.normalize(data_float.loc[:, 0], data_float.loc[:, 1], init=x_pos_min)

# Save sound
wav_name = path[:-4] + '_sound.wav'
path_mp3 = path[:-4] + '_sound.mp3'
x_pos_min = 1
_simplesound.save_sound(wav_name, data_float.loc[:,0], y1, init=x_pos_min) 
wav_to_mp3(wav_name, path_mp3)
