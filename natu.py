# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 08:30:39 2024

@author: JMCasado
"""

#General import
import os
import argparse
import glob
import numpy as np
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import pandas as pd

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

# Generate the plot if needed
ax.set_xlabel('x')
ax.set_ylabel('y')
# Separate the name file from the path to set the plot title
filename = os.path.basename(path)
#Plot 
ax.plot(data_float.loc[:, 0], data_float.loc[:, 1], 'b', linewidth=3)
ax.axhline(y=0, color='k', linewidth=1)
ax.axvline(x=0, color='k', linewidth=1)
# Set the path to save the plot and save it
plot_path = path[:-4] + 'plot.png'
fig.savefig(plot_path)

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
ax_noise.plot(data_float.loc[:, 0], y1_noise, 'r', linewidth=3)
ax_noise.set_xlabel('x')
ax_noise.set_ylabel('y')
ax_noise.axhline(y=0, color='k', linewidth=1)
ax_noise.axvline(x=0, color='k', linewidth=1)
image_name = path[:-4] + 'plot_noise.png'
fig_noise.savefig(image_name)