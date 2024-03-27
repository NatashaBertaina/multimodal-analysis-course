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

#Local import
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
_simplesound.reproductor.set_waveform('sine') # piano; sine
_simplesound.reproductor.set_time_base(0.2)
_simplesound.reproductor.set_min_freq(350)
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
ext = args.file_type or 'csv'
path = args.directory
plot_flag = args.save_plot or True

# Print a messege if path is not indicated by the user
if not path:
    print('1At least on intput must be stated.\nUse -h if you need help.')
    exit()

# Format the extension to use it with glob
extension = '*.' + ext

# Initialize a counter to show a message during each loop
i = 1
if plot_flag:
    # Create an empty figure or plot to save it
    cm = 1/2.54  # centimeters in inches
    #fig = plt.figure(figsize=(15*cm, 10*cm), dpi=300)
    fig = plt.figure()
    # Defining the axes so that we can plot data into it.
    #ax = plt.axes()
#Inits to generalize

# Loop to walk the directory and sonify each data file
now = datetime.datetime.now()
print(now.strftime('%Y-%m-%d_%H-%M-%S'))

# Open each file
data1, status, msg = _dataimport.set_arrayfromfile(path, ext)

# Convert into numpy, split in x and y and normalyze
print(msg)
print(data1)
if data1.shape[1]<2:
    print("Error reading file 1, only detect one column.")
    exit()
print(msg)


# Extract the names and turn to float
data_float1 = data1.iloc[1:, :].astype(float)

print(data_float1)



#Inicializamos la posición para no tener error cuando no haya recortes
x_pos_min = 0


# Generate the plot if needed
if plot_flag:
    
    #First plot
    #ax1 = plt.subplot(311)      
    ax = plt.subplot()    #para un solo plot
    ax.plot(data_float1.loc[:,0], data_float1.loc[:,1], 'b', linewidth=2)
    ax.tick_params('x', labelbottom=False)
    
    ax.legend()

    plt.pause(0.05)
    # Set the path to save the plot and save it
    plot_path = path[:-6] + 'plot.png'
    fig.savefig(plot_path)
    
# Reproduction
# Normalize the data to sonify
x1, y1, status = _math.normalize(data_float1.loc[:,0], data_float1.loc[:,1], init=x_pos_min)


# Reproduction
minval1 = float(data_float1.loc[:,1].min())
maxval1 = float(data_float1.loc[:,1].max())


input("Press Enter to continue...")


# Save sound
wav_name = path[:-4] + '_sound.wav'
path_mp3 = path[:-4] + '_sound.mp3'

x_pos_min = 1
_simplesound.save_sound(wav_name, data_float1.loc[:,0], y1, init=x_pos_min) 
wav_to_mp3(wav_name, path_mp3)

now = datetime.datetime.now()
print(now.strftime('%Y-%m-%d_%H-%M-%S'))
