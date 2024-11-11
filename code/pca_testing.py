# -*- coding: utf-8 -*-
"""
Testing PCA analysis on the images. 

Created on Wed Nov  6 16:09:11 2024

@author: janem
"""
# %% Imports

# Analysis
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Plotting
import matplotlib.gridspec as GS
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cm'
from matplotlib.colors import ListedColormap

import kaggle_helpers as kh
import pickle

# %% Getting data
print('Loading in the training data...')
kpath = './understanding_cloud_organization'  # path to images

# Opening the better df pkl located in the same directory as this file
with open('better_df.pkl', 'rb') as f:
    label_keys = pickle.load(f)

# %% Investigating a random image
print('Selecting a random image with all 4 masks...')
class_names = ['Fish', 'Flower', 'Gravel', 'Sugar']
img_df = label_keys[label_keys.num_labels == len(class_names)].sample(1).iloc[0]
img = kh.get_img(img_df.im_id, kpath)
ht, wd, n_clrs = img.shape  # height and width of the images
N_px = int(ht * wd)         # Number of pixels per image

# %% Trying out PCA on the images
print('Applying PCA...')
X = img.reshape(-1, n_clrs)
pca = PCA(n_components=n_clrs)
pca.fit(X)
img_PCA = pca.fit_transform(X)
img_PCA = np.reshape(img_PCA, (ht, wd, n_clrs))
img_PCA = kh.norm_matrix(img_PCA)

# %% Trying out k-means on the image
print('Applying K-Means...')
K = 4
kmeans = KMeans(n_clusters=K).fit(X)
kmeans_labels = kmeans.predict(X)
img_kmeans = np.reshape(kmeans_labels, (ht, wd))

# %% Showing images side-by-side
print('Plotting...')
fig = plt.figure(figsize=(7.5, 4), layout='constrained')
gs = GS.GridSpec(2, 1, figure=fig)
top_gs = GS.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0],
                                    hspace=0)
bottom_gs = GS.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0],
                                       hspace=0)
colors = ['c', 'm', 'y', 'g']

# --- Plotting original image --- #
ax = fig.add_subplot(top_gs[0, 0])
ax.imshow(img / 255)                        # Plotting img
# Plotting masks
for class_name, color in zip(class_names, colors):
    mask_rle = img_df[f'{class_name}']
    mask_img = kh.rle2mask(mask_rle)
    ax.contour(mask_img, colors=color, linewidths=0.3)
    cmap = ListedColormap(['w', color])
    ax.imshow(mask_img, alpha=mask_img * 0.1, cmap=cmap)
# Formatting
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f'Original')

# --- Plotting K-means --- #
ax = fig.add_subplot(top_gs[0, 1])
ax.imshow(img_kmeans)                        # Plotting img
# Plotting masks
for class_name, color in zip(class_names, colors):
    mask_rle = img_df[f'{class_name}']
    mask_img = kh.rle2mask(mask_rle)
    ax.contour(mask_img, colors=color, linewidths=0.3)
    cmap = ListedColormap(['w', color])
    ax.imshow(mask_img, alpha=mask_img * 0.1, cmap=cmap)
# Formatting
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f'K-Means (K={K:.0f})')

# --- Plotting PCA --- #
for col in range(n_clrs):
    ax = fig.add_subplot(bottom_gs[0, col])
    ax.imshow(img_PCA[:, :, col], cmap='gray')
    # Plotting masks
    for class_name, color in zip(class_names, colors):
        mask_rle = img_df[f'{class_name}']
        mask_img = kh.rle2mask(mask_rle)
        ax.contour(mask_img, colors=color, linewidths=0.3)
        cmap = ListedColormap(['w', color])
        ax.imshow(mask_img, alpha=mask_img * 0.1, cmap=cmap)
    # Formatting
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 1:
        ax.set_title('PCA')
fig.suptitle(f'Data Reduction with {img_df.im_id}')
savefig('../figs/data_redux.pdf', bbox_inches='tight', dpi=600)
plt.show()
