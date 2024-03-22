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
import matplotlib.pyplot as plt
import math
import pandas as pd

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
_simplesound.reproductor.set_waveform('celesta')
_simplesound.reproductor.set_time_base(0.05)
_simplesound.reproductor.set_min_freq(300)
_simplesound.reproductor.set_max_freq(1500)
# The argparse library is used to pass the path and extension where the data
# files are located
parser = argparse.ArgumentParser()
# Receive the extension from the arguments
parser.add_argument("-t", "--file-type", type=str,
                    help="Select file type.",
                    choices=['csv', 'txt'])
# Receive the directory path from the arguments
parser.add_argument("-d1", "--directory1", type=str,
                    help="Indicate a directory to process as batch.")
parser.add_argument("-d2", "--directory2", type=str,
                    help="Indicate a directory to process as batch.")
# Indicate to save or not the plot
parser.add_argument("-p", "--save-plot", type=bool,
                    help="Indicate if you want to save the plot (False as default)",
                    choices=[False,True])
# Alocate the arguments in variables, if extension is empty, select txt as
# default
args = parser.parse_args()
ext = args.file_type or 'csv'
#path = args.directory
path1 = args.directory1
path2 = args.directory2
plot_flag = args.save_plot or True
# Print a messege if path is not indicated by the user
if not path1:
    print('1At least on intput must be stated.\nUse -h if you need help.')
    exit()
if not path2:
    print('2At least on intput must be stated.\nUse -h if you need help.')
    exit()
# Format the extension to use it with glob
extension = '*.' + ext

# Initialize a counter to show a message during each loop
i = 1
if plot_flag:
    # Create an empty figure or plot to save it
    #cm = 1/2.54  # centimeters in inches
    #fig = plt.figure(figsize=(5*cm, 1*cm), dpi=300)
    fig = plt.figure()
    # Defining the axes so that we can plot data into it.
    ax = plt.axes()
#Inits to generalize

# Loop to walk the directory and sonify each data file
now = datetime.datetime.now()
print(now.strftime('%Y-%m-%d_%H-%M-%S'))

#for filename in glob.glob(os.path.join(path, extension)):
print("Converting data file number "+str(i)+" to sound.")
# Open each file
data1, status, msg = _dataimport.set_arrayfromfile(path1, ext)
data2, status, msg = _dataimport.set_arrayfromfile(path2, ext)
# Convert into numpy, split in x and y and normalyze
if data1.shape[1]<2:
    print("Error reading file 1, only detect one column.")
    exit()
if data2.shape[1]<2:
    print("Error reading file 2, only detect one column.")
    exit()
# Extract the names and turn to float
data_float1 = data1.iloc[1:, :].astype(float)
data_float2 = data2.iloc[1:, :].astype(float)

# Cut first data set
abs_val_array = np.abs(data_float1.loc[:,0] - 6800)
x_pos_min = abs_val_array.idxmin()
abs_val_array = np.abs(data_float1.loc[:,0] - 7200)
x_pos_max = abs_val_array.idxmin()
#data_float1 = data1.iloc[x_pos_min:x_pos_max, :].astype(float)
# Cut second data set
#data_float2 = data2.iloc[x_pos_min:x_pos_max, :].astype(float)

# Generate the plot if needed
if plot_flag:
    # Configure axis, plot the data and save it
    # Erase the plot
    ax.cla()
    # First file of the column is setted as axis name
    x_name = str(data1.iloc[0,0])
    ax.set_xlabel(x_name)
    # Separate the name file from the path to set the plot title
    #head, tail = os.path.split(filename)
    ax.plot(data_float1.loc[:,0], data_float1.loc[:,1], label='Flux-Barred spiral')
    #ax.plot(data_float1.loc[:,0], data_float1.loc[:,2], label='Best Fit')
    #ax.plot(data_float1.loc[:,0], data_float1.loc[:,3], label='Sky Flux')
    ax.plot(data_float2.loc[:,0], data_float2.loc[:,1], label='Flux-Double nucleus')
    ax.legend()
    # Set the path to save the plot and save it
    plot_path = path1[:-4] + '_plot.png'
    fig.savefig(plot_path)
    plt.pause(0.001)
# Normalize the data to sonify
x1, y1, status = _math.normalize(data_float1.loc[:,0], data_float1.loc[:,1], init=1)
#x, y2, status = _math.normalize(data_float1.loc[:,0], data_float1.loc[:,2], init=1)
#x, y3, status = _math.normalize(data_float1.loc[:,0], data_float1.loc[:,3], init=1)
x2, y2, status = _math.normalize(data_float2.loc[:,0], data_float2.loc[:,1], init=1)

# Reproduction
minval1 = float(np.abs(data_float1.loc[:,1]).min())
maxval1 = float(np.abs(data_float1.loc[:,1]).max())
minval2 = float(data_float2.loc[:,1].min())
maxval2 = float(data_float2.loc[:,1].max())

ordenada = np.array([min(minval1, minval2), max(maxval1, maxval2)])

input("Press Enter to continue...")

for x in range (1, data_float1.loc[:,0].size):
    # Plot the position line
    if not x == 1:
        line = red_line.pop(0)
        line.remove()
    abscisa = np.array([float(data_float1.loc[x,0]), float(data_float1.loc[x,0])])
    red_line = ax.plot(abscisa, ordenada, 'r')
    plt.pause(0.05)
    # Make the sound
    _simplesound.reproductor.set_waveform('sine')
    _simplesound.make_sound(y1[x], 1)
    _simplesound.reproductor.set_waveform('flute')
    _simplesound.make_sound(y2[x], 1)

# Save sound
wav_name = path1[:-4] + '_sound.wav'
#_simplesound.save_sound_multicol(wav_name, data_float1.loc[:,0], y1, y2, init=1)
# Print time
now = datetime.datetime.now()
print(now.strftime('%Y-%m-%d_%H-%M-%S'))
# Counter
i = i + 1

plt.pause(0.5)
# Showing the above plot
plt.show()
plt.close()
