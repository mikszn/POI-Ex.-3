from PIL import Image
from skimage.color import rgb2gray
from scipy.interpolate import interp1d
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import os
import pandas as pd

import skimage.io as io

def crop_image(source, target_dir, target_name, sample_size):
    im = Image.open(source)

    # Liczba fragmentów do wycięcia
    crops = [(im.size[0] // sample_size), (im.size[1] // sample_size)]

    idx = 1
    for row in range(crops[0]): # od 0 do 15
        for col in range(crops[1]):
                   #(left,            upper,           right,                         upper                        )
            im.crop((row*sample_size, col*sample_size, row*sample_size + sample_size, col*sample_size + sample_size)).save((target_dir+target_name+'_{num:d}'+'.jpg').format(num = idx), quality=95)
            idx += 1

def analyze_dir_of_images(source):
    matrix = []
    matrix_initialized = False

    for filename in os.listdir(source):
        f = os.path.join(source, filename)
        if os.path.isfile(f):

            im = rgb2gray(Image.open(f)) # konwersja do skali szarości
            m = interp1d([0,1], [0,63])
            im = m(im).astype(int) # Przemapowanie do 5 bitów
            glcm= graycomatrix(im, distances=[1,3,5], angles=[0,45,90,135], levels = 64, symmetric=True, normed=True)

            # 1-0, 1-45, 1-90, 1-135, 3-0, 3-45, 3-90, 3-135, 5-0...
            diss = graycoprops(glcm, 'dissimilarity').reshape(-1)
            corr = graycoprops(glcm, 'correlation').reshape(-1)
            cont = graycoprops(glcm, 'contrast').reshape(-1)
            ener = graycoprops(glcm, 'energy').reshape(-1)
            homo = graycoprops(glcm, 'homogeneity').reshape(-1)
            asm = graycoprops(glcm, 'ASM').reshape(-1)

            vector = np.concatenate((diss, corr, cont, ener, homo, asm))
            vector = np.append(vector, source)
            if not matrix_initialized:
                matrix = vector
                matrix_initialized = True
            else:
                matrix = np.vstack([matrix, vector])

    return matrix

def analyze_all_images():

    matrix = []

    # Nagłówek
    names = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM', 'Category']
    distances = ['1','3','5']
    angles = ['0', '45', '90', '135']

    for i in range(73):
        if i != 72:
            name = names[i//12]+'_'+distances[(i%12)//4]+'_'+angles[i%4]+'deg'
        else:
            name = names[i//12]
        matrix = np.append(matrix, name)

    matrix = np.vstack([matrix, analyze_dir_of_images('Laminat')])
    matrix = np.vstack([matrix, analyze_dir_of_images('Gres')])
    matrix = np.vstack([matrix, analyze_dir_of_images('Tkanina')])
    matrix = np.vstack([matrix, analyze_dir_of_images('Tynk')])

    pd.DataFrame(matrix).to_csv('textures_data.csv', sep=' ', header=False, index=False)


# Wycinanie fragmentów obrazow źródłowych
# crop_image('Gres/Oryginal/gres.jpg', 'Gres/', 'gres', 256)
# crop_image('Laminat/Oryginal/laminat.jpg', 'Laminat/', 'laminat', 256)
# crop_image('Tkanina/Oryginal/tkanina.jpg', 'Tkanina/', 'tkanina', 256)
# crop_image('Tynk/Oryginal/tynk.jpg', 'Tynk/', 'tynk', 256)

analyze_all_images()
