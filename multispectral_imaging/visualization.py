import rasterio
import numpy as np
import matplotlib.pyplot as plt
from msi_processing import *

with rasterio.open('Altum_PT_Ortho.tif') as data:
    rgb_data = []
    for i in range(3,0,-1):
        band = data.read(i)
        rgb_data.append(normalize_data(band))

    rgb_img = np.dstack(tuple(rgb_data))
    del rgb_data

    re_data = normalize_data(data.read(4))
    nir_data = normalize_data(data.read(5))

    f = plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(rgb_img)
    plt.title('RGB')
    plt.axis('off')
    del rgb_img

    plt.subplot(1,3,2)
    plt.imshow(re_data, cmap='Reds')
    plt.title('RE')
    plt.axis('off')
    del re_data

    plt.subplot(1,3,3)
    plt.imshow(nir_data, cmap='Spectral_r')
    plt.title('NIR')
    plt.axis('off')
    del nir_data

    plt.savefig("visualization.png")