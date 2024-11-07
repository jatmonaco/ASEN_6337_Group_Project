# -*- coding: utf-8 -*-
"""
Takes the training data and does the following to each image: 
    * Performs PCA analysis
    * Downscales the PCA image 
    * Writes out the downscaled PCA image to a folder 
    * Downscales the masks 
    * Writes out the downscaled mask to a new dataframe 

Created on Thu Nov  7 15:54:39 2024

@author: J. Monaco
"""
# %% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kaggle_helpers as kh
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cm'
import pickle
import tqdm

import cv2
from PIL import Image
from sklearn.decomposition import PCA
from skimage.transform import resize

# %% Loading in the image data
print('Loading in the training data...')
kpath = './understanding_cloud_organization'     # path to images
with open('better_df.pkl', 'rb') as f:
    label_keys = pickle.load(f)
labels = ['Sugar', 'Flower', 'Gravel', 'Fish']

# Getting info about pictures by investigating a random image
rand_img_df = label_keys.sample(1).iloc[0]
rand_img = kh.get_img(rand_img_df.im_id, kpath)
ht, wd, n_clrs = rand_img.shape  # height and width of the images

# %% PCA and downscale
input_folder = kpath + '/train_images'
output_folder = './downscaled_imgs'
scale_factor = 2
for idx, img_df in tqdm.tqdm(label_keys.iterrows(), total=label_keys.shape[0]):
    # Getting image
    image_name = img_df.im_id
    image_path = os.path.join(input_folder, image_name)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # PCA data reduction
    ht, wd, n_clrs = img.shape
    X = img.reshape(-1, n_clrs)
    pca = PCA(n_components=1)
    pca.fit(X)
    img_PCA = pca.fit_transform(X)
    img_PCA = np.reshape(img_PCA, (ht, wd))
    img_PCA = kh.norm_matrix(img_PCA)

    # Downscaling the image and making sure it fits the conv. kernels
    ht_scaled = ht // scale_factor
    ht_scaled += 4 - (ht_scaled % 4)
    wd_scaled = wd // scale_factor
    wd_scaled += 4 - (wd_scaled % 4)
    img_PCA = resize(img_PCA,
                     (ht_scaled, wd_scaled))

    # Saving the images
    img_PCA = (img_PCA * 255).astype(np.uint8)
    img_out = Image.fromarray(img_PCA)
    img_name = img_df.im_id[0:-4]
    img_out.save(output_folder + f'/{img_name}.png', 'PNG')

    # Downscaling masks
    for label in labels:
        # Getting the mask for this row
        rle = img_df[f'{label}']

        # Downscaling the mask
        mask = kh.rle2mask(rle, shape=(ht, wd))
        mask = resize(mask,
                      (ht_scaled, wd_scaled))
        mask = np.round(mask)

        # Converting to rle
        rle = kh.mask2rle(mask)

        # writing it to the dataframe
        label_keys.loc[idx, f'label'] = rle

# %% Saving the dataframe 
with open(f'{output_folder}/downscaled_df.pkl', 'wb') as f:
    # Dump the data into the file
    pickle.dump(label_keys, f)
