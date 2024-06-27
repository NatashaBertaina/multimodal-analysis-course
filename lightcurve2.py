import pandas as pd
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from data_transform.predef_math_functions import PredefMathFunctions
from sound_module.simple_sound import simpleSound

# Instanciate the sonoUno clases needed
_simplesound = simpleSound()
_math = PredefMathFunctions()
# Sound configurations, predefined at the moment
_simplesound.reproductor.set_continuous()
_simplesound.reproductor.set_waveform('sine')  # piano; sine
_simplesound.reproductor.set_time_base(0.03)
_simplesound.reproductor.set_min_freq(380)
_simplesound.reproductor.set_max_freq(800)

parser = argparse.ArgumentParser()
# Receive the directory path from the arguments
parser.add_argument("-d", "--directory", type=str,
                    help="Indicate a directory to process as batch.")
# Indicate the star type
parser.add_argument("-s", "--star-type", type=str,
                    help="Indicate the star type to plot (RWPhe, V0748Cep, ZLep, CGCas, HWPup, MNCam)",
                    choices=['RWPhe', 'V0748Cep', 'ZLep', 'CGCas', 'HWPup', 'MNCam'])
args = parser.parse_args()
path = args.directory
starType = args.star_type

data = pd.read_csv(path)
toplot = data.copy()

# Constantes de algunas estrellas variables
if starType == 'CGCas':
    """https://asas-sn.osu.edu/variables/753bdd73-38a7-5e43-b6c0-063292c7f28d"""
    periodo = 4.3652815
    t0 = 2457412.70647
    BPRP = 0.719
elif starType == 'RWPhe':
    """https://asas-sn.osu.edu/variables/dfa51488-c6b7-5a03-abd4-df3c28273250"""
    periodo = 5.4134367
    t0 = 2458053.49761
    BPRP = 0.586
elif starType == 'V0748Cep':
    """https://asas-sn.osu.edu/variables/dfcbcf52-8a62-542f-9383-1c712d7c042c"""
    periodo = 2.5093526
    t0 = 2458024.93242
elif starType == 'ZLep':
    """https://asas-sn.osu.edu/variables/70cc7024-5027-52f9-a834-75c51f4a5064"""
    periodo = 0.9937068
    t0 = 2457699.6236
elif starType == 'MNCam':
    """https://asas-sn.osu.edu/variables/c3faa9d0-6e10-5775-8bb0-075defcd2578"""
    periodo = 8.1796049
    t0 = 2458046.08639
elif starType == 'HWPup':
    """https://asas-sn.osu.edu/variables/2083f661-73f5-512f-aee5-fd7ad26d5b30"""
    periodo = 13.4590914
    t0 = 2457786.63153
else:
    print('Error en el tipo de estrella.')

toplot['hjd'] = ((data['hjd'] - t0) / periodo)
toplot['hjd'] = (toplot['hjd'] - toplot['hjd'].astype(int))# + BPRP + 1

count = 0
for i in toplot['hjd']:
    if i < 0:
        toplot.loc[count,'hjd'] = i + 1
    count = count + 1

toplot['hjd'] = toplot['hjd'] + BPRP

count = 0
for i in toplot['hjd']:
    if i > 1:
        toplot.loc[count,'hjd'] = i - 1
    count = count + 1
toplotlength = count

for i in toplot['hjd']:
    toplot.loc[count,'hjd'] = i + 1
    toplot.loc[count, 'camera'] = toplot.loc[count-toplotlength, 'camera']
    toplot.loc[count, 'mag'] = toplot.loc[count-toplotlength, 'mag']
    toplot.loc[count, 'mag_err'] = toplot.loc[count-toplotlength, 'mag_err']
    toplot.loc[count, 'flux'] = toplot.loc[count-toplotlength, 'flux']
    toplot.loc[count, 'flux_err'] = toplot.loc[count-toplotlength, 'flux_err']
    count = count + 1

groups = toplot.groupby('camera')
if starType == 'CGCas':
    bd_toplot = groups.get_group('bd')
    bc_toplot = groups.get_group('bc') 
elif starType == 'RWPhe':
    be_toplot = groups.get_group('be')
    bf_toplot = groups.get_group('bf') 
else:
    print('Error en el tipo de estrella para separar por grupos.')

fig = plt.figure()
ax = plt.axes()

ax.set_xlabel('Phase')
ax.set_ylabel('Mag')
ax.invert_yaxis()
if starType == 'CGCas':
    ax.set_title('CG-Cas-Cepheid')
    ax.scatter(bd_toplot['hjd'], bd_toplot['mag'], marker='.', c='#CF3476', label='bd')
    ax.scatter(bc_toplot['hjd'], bc_toplot['mag'], marker='.', c='#E59866', label='bc')
    plot_path = 'data/galaxy-stars/light-curves/Cefeida/CGCas/cepheid_sonouno.png'

    bd_toplot = pd.DataFrame(bd_toplot).sort_values(0, axis=0)
    bc_toplot = pd.DataFrame(bc_toplot).sort_values(0, axis=0)

    print(bd_toplot)

    x_bd, y_bd, status = _math.normalize(bd_toplot['hjd'], bd_toplot['mag'])
    x_bc, y_bc, status = _math.normalize(bc_toplot['hjd'], bc_toplot['mag'])

    ordenada = np.array([bd_toplot['mag'].min(), bd_toplot['mag'].max()])
    """for x in range (0, bd_toplot['hjd'].size+bc_toplot['hjd'].size):
        # Plot the position line
        if not x == 0:
            line = red_line.pop(0)
            line.remove()
        status = True
        #abscisa = np.array([float(data_float1.loc[x,0]), float(data_float1.loc[x,0])])
        try:
            status = False
            abscisa = np.array([bd_toplot['hjd'][x], bd_toplot['hjd'][x]])
            red_line = ax.plot(abscisa, ordenada, 'r')
            plt.pause(0.05)
            # Make the sound
            _simplesound.reproductor.set_waveform('sine')
            _simplesound.make_sound(y_bd[x], 1)
        except Exception as e:
            #print(e)
            pass
        try:
            status = False
            abscisa = np.array([bc_toplot['hjd'][x], bc_toplot['hjd'][x]])
            red_line = ax.plot(abscisa, ordenada, 'r')
            plt.pause(0.05)
            # Make the sound
            _simplesound.reproductor.set_waveform('flute')
            _simplesound.make_sound(y_bc[x], 1)
        except Exception as e:
            #print(e)
            pass
        if status:
            print('no entró a ningún try')"""

    #wav_path = 'data/galaxy-stars/light-curves/Cefeida/CGCas/cepheid_sonouno.wav'
    #_simplesound.save_sound(wav_path, x_bd, y_bd)
elif starType == 'RWPhe':
    ax.set_title('RW-Phe-Eclipsing Binary')
    ax.scatter(be_toplot['hjd'], be_toplot['mag'], marker='.', c='k', label='be')
    ax.scatter(bf_toplot['hjd'], bf_toplot['mag'], marker='.', c='g', label='bf')
    plot_path = 'data/galaxy-stars/light-curves/BEclipsante/RWPhe/eclipsante_sonouno.png'
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

ax.legend()    
# Save the plot
fig.savefig(plot_path)

plt.pause(0.5)
# Showing the above plot
plt.show()
plt.close()