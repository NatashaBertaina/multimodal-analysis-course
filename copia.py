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
data, status, msg = _dataimport.set_arrayfromfile(path, ext)

# Convert into numpy, split in x and y and normalyze
if data.shape[1]<2:
    print("Error reading file 1, only detect one column.")
    exit()

# Extract the names and turn to float
data_float = data.iloc[1:, :].astype(float)
x_pos_min = 1

# Generate the plot if needed
if plot_flag:
    # Configure axis, plot the data and save it
    # Erase the plot
    #ax.cla()
    # First file of the column is setted as axis name
    #x_name = str(data1.iloc[0,0])
    #ax.set_xlabel(x_name)
    # Separate the name file from the path to set the plot title
    #head, tail = os.path.split(filename)
    
    #Plot   
    ax = plt.subplot()    #para un solo plot
    ax.plot(data_float.loc[:,0], data_float.loc[:,1])
    ax.tick_params('x', labelbottom=False)
    
    ax.legend()

    plt.pause(0.05)
    # Set the path to save the plot and save it
    plot_path = path[:-4] + 'plot.png'
    fig.savefig(plot_path)
    
# Reproduction
# Normalize the data to sonify
x1, y1, status = _math.normalize(data_float.loc[:,0], data_float.loc[:,1], init=x_pos_min)

# Cut x axis
minval = float(data_float.loc[:,1].min())
maxval = float(data_float.loc[:,1].max())

# To make reproduction on real time
ordenada = np.array([minval,maxval])

#input("Press Enter to continue...")

#for i in range (1, 4):
    #print(i)
    #for x in range (x_pos_min, x_pos_max):
        #if i==1:
            # Plot the position line
            #if not x == x_pos_min:
                #line = red_line.pop(0)
                #line.remove()
            #abscisa = np.array([float(data_float1.loc[x,1]), float(data_float1.loc[x,1])])
            #red_line = ax1.plot(abscisa, ordenada1, 'r')
            #plt.pause(0.05)
            # Make the sound
            #_simplesound.reproductor.set_waveform('sine')
            #_simplesound.make_sound(y1[x], 1)
            #_simplesound.reproductor.set_waveform('flute')
            #_simplesound.make_sound(y4[x], 1)
            #if x == (x_pos_max-1):
                #line = red_line.pop(0)
                #line.remove()
        #if i==2:
            # Plot the position line
            #if not x == x_pos_min:
                #line = red_line.pop(0)
                #line.remove()
            #abscisa = np.array([float(data_float1.loc[x,1]), float(data_float1.loc[x,1])])
            #red_line = ax2.plot(abscisa, ordenada2, 'r')
            #plt.pause(0.05)
            # Make the sound
            #_simplesound.reproductor.set_waveform('sine')
            #_simplesound.make_sound(y2[x], 1)
            #_simplesound.reproductor.set_waveform('flute')
            #_simplesound.make_sound(y4[x], 1)
            #if x == (x_pos_max-1):
                #line = red_line.pop(0)
                #line.remove()
        #if i==3:
            # Plot the position line
            #if not x == x_pos_min:
                #line = red_line.pop(0)
                #line.remove()
            #abscisa = np.array([float(data_float1.loc[x,1]), float(data_float1.loc[x,1])])
            #red_line = ax3.plot(abscisa, ordenada3, 'r')
            #plt.pause(0.05)
            # Make the sound
            #_simplesound.reproductor.set_waveform('sine')
            #_simplesound.make_sound(y3[x], 1)
            #_simplesound.reproductor.set_waveform('flute')
            #_simplesound.make_sound(y4[x], 1)
            #if x == (x_pos_max-1):
                #line = red_line.pop(0)
                #line.remove()

# Save sound
wav_name = path[:-4] + '_sound.wav'
path_mp3 = path[:-4] + '_sound.mp3'
x_pos_min = 1
_simplesound.save_sound(wav_name, data_float.loc[:,0], y1, init=x_pos_min) 
wav_to_mp3(wav_name, path_mp3)

# Print time
now = datetime.datetime.now()
print(now.strftime('%Y-%m-%d_%H-%M-%S'))

plt.pause(0.5)
# Showing the above plot
plt.show()
plt.close()
