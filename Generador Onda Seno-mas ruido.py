import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

default_amplitude = 4096 # Asociado al volumen

# Definiendo la onda seno
def get_sine_wave(frequency, duration, sample_rate=44100, amplitude=default_amplitude):
    t = np.linspace(0, duration, int(sample_rate*duration)) # Time axis
    wave = amplitude*np.sin(2*np.pi*frequency*t)
    return wave

# Obtener onda seno pura
sine_wave_1 = get_sine_wave(300, duration=2)
sine_wave_2 = get_sine_wave(305, duration=2)
sine_wave_3 = get_sine_wave(310, duration=2)
sine_wave_4 = get_sine_wave(315, duration=2)

# Sumar todas las ondas senos obtenidas para generar una única onda
sound = np.append(sine_wave_1, sine_wave_2)
sound = np.append(sound, sine_wave_3)
sound = np.append(sound, sine_wave_4)


# Creación archivo wav de onda seno limpia
wavfile.write('sound.wav', rate=48000, data=sound.astype(np.int16))

# Grafico señal limpia
# Extraer valores
audio = sound[:2000]

# Construyendo el eje del tiempo
x_values = np.arange(0, len(audio), 1) / float(44100)

# Convirtiendo a segundos
x_values *= 1000

# Graficando la señal
plt.plot(x_values, audio, color='red')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
#plt.title('Señal de audio')
plt.show()

#Módulo de ruido
audio_watts = sound ** 2

target_snr_db = 1 #SNR objetivo

sig_avg_watts = np.mean(audio_watts) # Calcular potencia media
sig_avg_db = 20 * np.log10(sig_avg_watts) # Obtener valor en dB del sonido

noise_avg_db = sig_avg_db - target_snr_db # Calcular dB de ruido en base del sonido sintetizado
noise_avg_watts = 10 ** (noise_avg_db / 10)

mean_noise = 0 # Generacion ruido blanco
noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(audio_watts))
sound += noise

wavfile.write('noise.wav', rate = 48000, data=noise.astype(np.int16))
wavfile.write('sound-noise.wav', rate = 48000, data=sound.astype(np.int16))

# Gráfico sonido mas ruido
# Extraer valores
audio = sound[:2000]

# Construyendo el eje del tiempo
x_values = np.arange(0, len(audio), 1) / float(44100)

# Convirtiendo a segundos
x_values *= 1000

# Graficando la señal
plt.plot(x_values, audio, color='black')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
#plt.title('Señal de audio')
plt.show()

