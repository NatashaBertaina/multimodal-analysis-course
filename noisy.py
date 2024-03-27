import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#Archivo de Salida
output_file = 'sonido_mas_ruido.wav'

#Especificando los parámetros del audio
duracion = 20  # segundos
sampling_freq = 44100  # Hz
tone_freq = 100
min_val = 1 * np.pi
max_val = 100 * np.pi

# Generar audio
t = np.linspace(min_val, max_val, duracion * sampling_freq) #arreglo de valores: numero inicial, numero final, cant de valores
audio = 10 * np.sin(2 * np.pi * tone_freq * t)
audio_watts = audio ** 2
x_db = 10 * np.log10(audio_watts)

t1= t [:250]
audio_plot = audio [:250]
##graficar sonido limpio
plt.subplot (3,1,1)
plt.plot (t1, audio_plot)
plt.title ("Señal Limpia")
plt.ylabel ("Amplitud")
plt.xlabel ("Tiempo (s)")
plt.show()


# Agregar ruido con un SNR deseado
#SNR objetivo
target_snr_db = 1

sig_avg_watts = np.mean(audio_watts) # Calcular potencia 
sig_avg_db = 10 * np.log10(sig_avg_watts)

noise_avg_db = sig_avg_db - target_snr_db # calcular ruido
noise_avg_watts = 10 ** (noise_avg_db / 10)

mean_noise = 0 # Generacion ruido blanco
ruido_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(audio_watts))
# Ruido montado en la señal original
audio += ruido_volts


# Escalar a 16 bits los valores enteros
factor_escala = pow(2,15) - 1
audio_normalizado = audio / np.max(np.abs(audio))
audio_escalado = np.int16(audio_normalizado * factor_escala)

# Escribir el archivo de salida
wavfile.write(output_file, sampling_freq, audio_escalado)


# Extraer los primeros 100 valores
audio_escalado = audio[:250]

# Construyendo el eje del tiempo
x_values = np.arange (0, len(audio_escalado), 1) / float(sampling_freq)

# Convirtiendo a segundos
x_values *= 1000
audio= audio[:250]

# Graficando la señal
plt.subplot(3,1,3)
plt.plot(x_values, audio, color='red')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.title('Señal de audio ruidosa')
plt.show()

