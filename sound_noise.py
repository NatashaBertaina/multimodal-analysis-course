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

# Function to generate Gaussian noise
def generate_gaussian_noise(length, snr):
    signal_power = 10 ** (snr / 10)
    noise_var= 1 / signal_power
    return np.random.normal(0, np.sqrt(noise_var), length)

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

# Generate the braille plot
figbraille = plt.figure()
axbraille = plt.axes()

# 3 valores de eje x en braille

abs_val_array = np.abs(data_float.loc[:,0] - data_float.loc[:,0].min())
x_pos_min = abs_val_array.idxmin()
middle = ((data_float.loc[:,0].max() - data_float.loc[:,0].min())/2) + data_float.loc[:,0].min()
abs_val_array = np.abs(data_float.loc[:,0] - middle)
x_pos_middle = abs_val_array.idxmin()
abs_val_array = np.abs(data_float.loc[:,0] - data_float.loc[:,0].max())
x_pos_max = abs_val_array.idxmin()

xinicio_text = str(int(data_float.loc[x_pos_min,0]))
xinicio_text = brl.translate(xinicio_text)
if data_float.loc[x_pos_min,0] < 0:
    caract_resta = [['001001']]
    for i in xinicio_text[0]:
        caract_resta[0].append(i)
    xinicio_text = caract_resta
xinicio_text = brl.toUnicodeSymbols(xinicio_text, flatten=True)
xmedio_text = str(int(data_float.loc[x_pos_middle,0]))
xmedio_text = brl.translate(xmedio_text)
if data_float.loc[x_pos_middle,0] < 0:
    caract_resta = [['001001']]
    for i in xmedio_text[0]:
        caract_resta[0].append(i)
    xmedio_text = caract_resta
xmedio_text = brl.toUnicodeSymbols(xmedio_text, flatten=True)
xfinal_text = str(int(data_float.loc[x_pos_max,0]))
xfinal_text = brl.translate(xfinal_text)
if data_float.loc[x_pos_max,0] < 0:
    caract_resta = [['001001']]
    for i in xfinal_text[0]:
        caract_resta[0].append(i)
    xfinal_text = caract_resta
xfinal_text = brl.toUnicodeSymbols(xfinal_text, flatten=True)
axbraille.set_xticks([data_float.loc[x_pos_min,0],data_float.loc[x_pos_middle,0],data_float.loc[x_pos_max,0]], 
                     [xinicio_text,xmedio_text,xfinal_text], 
                     fontsize=24,
                     fontfamily='serif',
                     fontweight='bold')
# 3 valores de eje y en braille
# Found min, middle, max possitions and values
abs_val_array = np.abs(data_float.loc[:,1] - data_float.loc[:,1].min())
y_pos_min = abs_val_array.idxmin()
middle = ((data_float.loc[:,1].max() - data_float.loc[:,1].min())/2) + data_float.loc[:,1].min()
abs_val_array = np.abs(data_float.loc[:,1] - middle)
y_pos_middle = abs_val_array.idxmin()
abs_val_array = np.abs(data_float.loc[:,1] - data_float.loc[:,1].max())
y_pos_max = abs_val_array.idxmin()

y_pos_min_text = str(int(data_float.loc[y_pos_min,1]))
y_pos_min_text = brl.translate(y_pos_min_text)
if data_float.loc[y_pos_min,1] < 0:
    caract_resta = [['001001']]
    for i in y_pos_min_text[0]:
        caract_resta[0].append(i)
    y_pos_min_text = caract_resta
y_pos_min_text = brl.toUnicodeSymbols(y_pos_min_text, flatten=True)
y_pos_middle_text = str(int(data_float.loc[y_pos_middle,1]))
y_pos_middle_text = brl.translate(y_pos_middle_text)
if data_float.loc[y_pos_middle,1] < 0:
    caract_resta = [['001001']]
    for i in y_pos_middle_text[0]:
        caract_resta[0].append(i)
    y_pos_middle_text = caract_resta
y_pos_middle_text = brl.toUnicodeSymbols(y_pos_middle_text, flatten=True)
y_pos_max_text = str(int(data_float.loc[y_pos_max,1]))
y_pos_max_text = brl.translate(y_pos_max_text)
if data_float.loc[y_pos_max,1] < 0:
    caract_resta = [['001001']]
    for i in y_pos_max_text[0]:
        caract_resta[0].append(i)
    y_pos_max_text = caract_resta
y_pos_max_text = brl.toUnicodeSymbols(y_pos_max_text, flatten=True)
axbraille.set_yticks([data_float.loc[y_pos_min,1],data_float.loc[y_pos_middle,1],data_float.loc[y_pos_max,1]], 
                     [y_pos_min_text,y_pos_middle_text,y_pos_max_text], 
                     fontsize=24,
                     fontfamily='serif',
                     fontweight='bold')

axbraille.set_title(' ')
x = brl.translate('x')
x = brl.toUnicodeSymbols(x, flatten=True)
axbraille.set_xlabel(x, fontsize=24, fontfamily='serif', fontweight='bold', labelpad=10)
y = brl.translate('y')
y = brl.toUnicodeSymbols(y, flatten=True)
axbraille.set_ylabel(y, fontsize=24, fontfamily='serif', fontweight='bold', labelpad=10, rotation=0)
axbraille.plot(data_float.loc[:, 0], data_float.loc[:, 1], '#2874a6', linewidth=3)
# Ejes de coordenadas
if data_float.loc[:, 0].min() < 0:
    axbraille.axvline(x=0, color='k', linewidth=1)
if data_float.loc[:, 1].min() < 0:
    axbraille.axhline(y=0, color='k', linewidth=1)
# Resize
figbraille.tight_layout()
# Save braille figure
brailleplot_path = path[:-4] + 'plot-braille.png'
figbraille.savefig(brailleplot_path)
plt.close()

# Reproduction
# Normalize the data to sonify
x1, y1, status = _math.normalize(data_float.loc[:, 0], data_float.loc[:, 1], init=x_pos_min)

# Save sound
wav_name = path[:-4] + '_sound.wav'
path_mp3 = path[:-4] + '_sound.mp3'
x_pos_min = 1
_simplesound.save_sound(wav_name, data_float.loc[:,0], y1, init=x_pos_min) 
wav_to_mp3(wav_name, path_mp3)

# Generate sound with Gaussian noise
y1_noise = y1 + generate_gaussian_noise(len(y1), noise_snr) 

# Save sound
wav_name_noise = path[:-4] + '_noise.wav'
path_mp3_noise = path[:-4] + '_noise.mp3'
_simplesound.save_sound(wav_name_noise, data_float.loc[:, 0], y1_noise, init=x_pos_min)
wav_to_mp3(wav_name_noise, path_mp3_noise)

# Generate image of sound with noise
fig_noise = plt.figure()
ax_noise = plt.axes()
ax_noise.plot(data_float.loc[:, 0], y1_noise, '#f39c12', linewidth=3)
ax_noise.set_xlabel('x')
ax_noise.set_ylabel('y')
if data_float.loc[:, 0].min() < 0:
    ax_noise.axvline(x=0, color='k', linewidth=1)
if y1_noise.min() < 0:
    ax_noise.axhline(y=0, color='k', linewidth=1)
image_name = path[:-4] + 'plot_noise.png'
fig_noise.savefig(image_name)
plt.close()
