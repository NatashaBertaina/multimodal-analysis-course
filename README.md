# Multimodal script for tools used on a course 

This development is powered by CONICET-Argentina, Universidad de Mendoza-Argentina and Universidad Tecnológica Nacional - Facultad Regional Mendoza.

<Img src="logos/logo_conicet.png" width="100"> <Img src="logos/logosX2.png" width="120"> <Img src="logos/UTN-logo(1).png" width="100"> 

# Description

This repository hosts the scripts used for data sonification using the sonoUno libraries and tactile models generation using pybrl and Farjo et al (2024) development. It includes a set of scripts that allow scientific data to be transformed into accessible sound and visual representations using labels in braille.

# Main Features

### 1. Sonification with Noise:
   - The `sound_noise.py` script enables the sonification of tabulated data in two columns, with the option to introduce controlled noise adjusted to a specific signal-to-noise ratio.

### 2. Sonification  of Star Spectra:
   - The `starts.py` script is designed for the sonification of star spectra, facilitating the analysis and multisensory representation of this type of scientific data.

### 3. Sonification  of Light Curves:
   - The `lightcurve.py` script allows the sonification of light curves downloaded from the ASAS-SN database.

### 4. Tactile models
   - Each script includes the integration of the `pybrl` library to modify the labels of matplotlib graph. Pybrl is used to translate graph texts into Braille, ensuring the accessibility of generated images that can be printed in 3D models.

# How to contribute to the software

All help is welcomed and needed! If you want to contribute contact us at sonounoteam@gmail.com 

# Report issues or a problems with the software

All people could report a problem opening an issue here on GitHub or contact us by email: sonounoteam@gmail.com

# Bibliography

Farjo, C., Casado, J., García, B. (2024). Transformation of images to 3D tactile models from open source software. Revista Mexicana de Astronomía y Astrofísica Serie de Conferencias (RMxAC), 57, 37-41. DOI: \url{https://doi.org/10.22201/ia.14052059p.2024.57.09}
