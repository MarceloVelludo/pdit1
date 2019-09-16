import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from skimage import feature
from skimage import filters
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

import ImageReader
import RGBToGrayScale

#dialog box para pegar o path da imagem.
root = tk.Tk()
root.withdraw()

#root.withdraw()
path = filedialog.askopenfilename()

#Lê a imagem || Exercício 1
image = ImageReader.readImage(path)
#Fim Exercício 1

#Transforma a Imagem em Monocromática || Exercício 2
graysScale = RGBToGrayScale.rgb2gray(image)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
fig.suptitle("Original para Escala de Cinza", ha='center', va='top')
fig.dpi = 125

ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(graysScale, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

fig.tight_layout()
plt.savefig("GrayScale.png")
#plt.show()
plt.close(fig)

#Fim Exercício 2

#Encontra as bordas da imagem monocromática,
#utilizando um filtro passa-alta || Exercício 3
#Filtro Sobel
edges_sobel = filters.sobel(graysScale)
#Filtro robert
edge_roberts = filters.roberts(graysScale)
#Filtro Canny com diferentes sigmas
edge_canny_s1 = feature.canny(graysScale, sigma=1)
edge_canny_s2 = feature.canny(graysScale, sigma=2)
edge_canny_s3 = feature.canny(graysScale, sigma=3)

fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(8, 4))
fig.dpi = 250
ax1[0].imshow(edges_sobel, cmap=plt.cm.gray)
ax1[0].set_title("Sobel")
ax1[1].imshow(edge_roberts, cmap=plt.cm.gray)
ax1[1].set_title("Robert")
ax2[0].imshow(edge_canny_s1, cmap=plt.cm.gray)
ax2[0].set_title("Canny sigma=1")
ax2[1].imshow(edge_canny_s2, cmap=plt.cm.gray)
ax2[1].set_title("Canny sigma=2")
ax2[2].imshow(edge_canny_s3, cmap=plt.cm.gray)
ax2[2].set_title("Canny sigma=3")

for a in ax1:
    a.axis('off')

fig.tight_layout()
plt.savefig("FiltrosCaixaAlta.png")
#plt.show()
plt.close(fig)


#Fim Exercício 3

#Binariza a imagem com as bordas detctadas
#transformando-a em preto e branco.|| Método de Otsu||
#Exercício 4
thresh = filters.threshold_otsu(edges_sobel)
binary = edges_sobel > thresh

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

fig.dpi = 100
ax.imshow(binary, cmap=plt.cm.gray)
ax.set_title("Binarização Otsu")
ax.axis('off')
fig.tight_layout()
plt.savefig("BinarizacaoOtsu.png")
#plt.show()
plt.close(fig)

#Fim Exercício 4

#Inicio dos exercicios 5,6,7,8
#Calcula a transformada de Hough da imagem binária||
#Exercício 5
#Seta uma precisão de 0,5 graus
tested_angles = np.linspace(-np.pi/2, np.pi/2, 360)
h, theta, d = hough_line(binary, theta=tested_angles)

fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(8, 4))

fig.dpi = 200
ax1[0].imshow(binary, cmap=plt.cm.gray)
ax1[0].set_title("Binarização Otsu")
ax1[0].axis('off')

ax1[1].imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=plt.cm.gray, aspect=1/1.5)
ax1[1].set_title('Transformada de Hough')
ax1[1].set_xlabel('Angulos (graus)')
ax1[1].set_ylabel('Distancia (pixels)')

#Encontrando os Picos e as retas || Exercício 6,7,8
ax2[0].imshow(binary, cmap=plt.cm.gray)
origin = np.array((0, binary.shape[1]))
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax2[0].plot(origin, (y0, y1), '-r')
ax2[0].set_xlim(origin)
ax2[0].set_ylim((image.shape[0], 0))
ax2[0].set_axis_off()
ax2[0].set_title('Linhas Detectadas')

#Fim Exercício 5,6,7,8

#Extra: Hough Probabilistico
lines = probabilistic_hough_line(binary, line_length=3,
                                 line_gap=3)
ax2[1].imshow(binary*0)
for line in lines:
    p0, p1 = line
    ax2[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax2[1].set_xlim((0, binary.shape[1]))
ax2[1].set_ylim((binary.shape[0], 0))
ax2[1].set_title('Hough Probabilístico')
ax2[1].set_axis_off()
fig.tight_layout()
plt.savefig("TransformadadeHough.png")
#plt.show()
plt.close(fig)

#Fim da Parte Extra.

#Gerando uma lista a partir das retas encontradas || Exercício 9
count = 0
table = PrettyTable()
table.field_names = ["Linha", "Origem <r,c>", "Destino <r,c>"]

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    count += 1
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    table.add_row([count, (dist - origin * np.cos(angle)), (y0, y1)])
print(table)
file = open("testfile.txt", "w")
file.write(str(table))
file.close()
425083

#Fim Exercício 9



exit()