from sklearn.decomposition import PCA
import numpy as np

chicken = np.genfromtxt('input/chicken_HYSPEC.csv', delimiter='\t')
chicken_hsi = chicken[1:]

pca = PCA(n_components=5)

chicken_pca = pca.fit_transform(chicken_hsi)
chicken_hsi_im = chicken_hsi.reshape(100, 100, -1)
chicken_pca_im = chicken_pca.reshape(100, 100, -1)
np.save('output/chicken_pca_im.npy', chicken_pca_im)
np.save('output/chicken_hsi_im.npy', chicken_hsi_im)

print('done')

