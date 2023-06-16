from PIL import Image
from skimage.color import rgb2gray
from scipy.interpolate import interp1d
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import os
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

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

#Analiza fragmentów
# analyze_all_images()

features = pd.read_csv('textures_data.csv', sep=' ', )

data = np.array(features)
X = data[:,:-1].astype('float64') # wydobycie wartości cech jako floaty
Y = data[:,-1]    # wydobycie nazw kategorii


# ======== Wizualizacja rozróżnialności cech poszczególnych próbek ========
x_transform = PCA(n_components=3)
Xt = x_transform.fit_transform(X)

blue = Y == 'Gres'
cyan = Y == 'Laminat'
red = Y == 'Tkanina'
magenta = Y == 'Tynk'


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xt[blue, 0], Xt[blue, 1], Xt[blue, 2], c='b')
ax.scatter(Xt[cyan, 0], Xt[cyan, 1], Xt[cyan, 2], c='c')
ax.scatter(Xt[red, 0], Xt[red, 1], Xt[red, 2], c='r')
ax.scatter(Xt[magenta, 0], Xt[magenta, 1], Xt[magenta, 2], c='m')

plt.show()

# ============================ Klasyfikacja ===============================

classifier = svm.SVC(gamma='auto') # parametr gamma steruje nieliniowością granicy

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33) # x_ to cechy, y_ to kategorie

classifier.fit(x_train, y_train) # budowanie granicy decyzyjnej (trenowanie klasyfikatora)
y_pred = classifier.predict(x_test) # predykcja zbioru testowego
acc = accuracy_score(y_test, y_pred) # ocena predykcji
print(acc)

cm = confusion_matrix(y_test, y_pred, normalize='true') # wynik względny od 0 do 1 (bo normalize=True)
print(cm) # Macierz pomyłek w terminalu (względna)

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show() # Macierz pomyłek w formie  graficznej (bezwzględna)
