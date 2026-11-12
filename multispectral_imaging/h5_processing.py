import h5py
from math import sqrt
import numpy as np
from numpy.linalg import matmul, eig, lstsq
from matplotlib import pyplot as plt
from h5_helpers import *

def train_PCA(img_num, pc_num=-1):
    data_path = "data/dataset.h5"

    with h5py.File(data_path, "r") as data:
        # reformat image to matrix of size pixels*wavelengths
        image = data["train"]["sentinel2"]["06_june_2019"][img_num]
        image_mat = reformat_image(image)

        # find principal components of image data
        pcs, vals = calculate_pc(image_mat, pc_num)
        visualize_eigenvalues(vals)
        visualize_PCs(pcs, 2)

        # project data onto principal components
        means = image_mat.mean(axis=0)
        mean_centered = image_mat - means
        projected = matmul(mean_centered, pcs)

        # reformat labels to column vector
        labels = data["train"]["labels"]["field"][img_num]
        labels_mat = reformat_labels(labels)

        # use projected data and labels to approximate weights for the data
        weights = lstsq(projected, labels_mat)[0]
        return weights, means


def reformat_image(image):
    return image.reshape((image.shape[0], image.shape[1] * image.shape[2])).transpose()

def reformat_labels(labels):
    labels_flat = flatten_labels(labels)
    return labels_flat.reshape((labels_flat.shape[0]*labels_flat.shape[1],1))

def calculate_pc(mat, num=-1):
    # check that num value is possible
    if num == -1:
        num = mat.shape[1]
    elif num > mat.shape[1]:
        raise ValueError("Number of possible principal components is less than number requested")

    # make correlation matrix
    means = mat.mean(axis=0)
    corr_A = (mat - means)/sqrt(mat.shape[0]-1)
    corr_R = matmul(corr_A.transpose(), corr_A)

    # get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(corr_R)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # order by decreasing eigenvalue
    indices = sorted(list(range(eigenvalues.shape[0])), key=lambda i: eigenvalues[i])[num-1::-1]
    sorted_eigenvalues = np.array([eigenvalues[idx] for idx in indices])
    sorted_eigenvectors = np.zeros((num,eigenvectors.shape[1]))
    for i in range(len(indices)):
        sorted_eigenvectors[:,i] = eigenvectors[:,indices[i]]
    
    # return sorted eigenvectors and eigenvalues as tuple
    return sorted_eigenvectors, sorted_eigenvalues

def visualize_eigenvalues(eigenvalues):
    plt.figure()
    plt.plot(list(range(1,eigenvalues.shape[0]+1)), eigenvalues, '.-')
    plt.title('Eigenvalues')
    plt.ylabel('Eigenvalue')
    plt.xlabel('Principal Component #')
    plt.savefig('graphics/eigenvalues.png')

def visualize_PCs(pcs, num=1):
    plt.figure()
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.plot(list(range(1,pcs.shape[0]+1)), pcs[:,i], '.-')
        plt.xlim(0.5, 10.5)
        plt.ylim(-0.5, 1)
        plt.title(f"Principal Component {i+1}")
        plt.xlabel("Weight")
        plt.ylabel('Wavelength')
    plt.savefig(f'graphics/principal_components.png')
