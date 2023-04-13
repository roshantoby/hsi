from sklearn.decomposition import PCA
import numpy as np


def run_pca(hsi_data, height, width):
    hsi_im = hsi_data[1:]

    pca = PCA(n_components=5)

    hsi_pca = pca.fit_transform(hsi_im)
    hsi_im = hsi_im.reshape(height, width, -1)
    hsi_pca = hsi_pca.reshape(height, width, -1)
    np.save('output/pca_im.npy', hsi_pca)
    np.save('output/hsi_im.npy', hsi_im)

    return None
