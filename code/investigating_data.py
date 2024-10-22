# -*- coding: utf-8 -*-
"""
Drawn from: https://www.kaggle.com/code/artgor/segmentation-in-pytorch-using-convenient-tools
Created on Tue Oct 22 12:52:27 2024

@author: J. Monaco
"""
# %% Imports
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# %% Getting data
path = './understanding_cloud_organization'
print(f'Files and Folders available: {os.listdir(path)}')

# Number of files
N_labeled = len(os.listdir(f'{path}/train_images'))
N_test = len(os.listdir(f'{path}/test_images'))
print(f'{N_labeled} training images and {N_test} test images available')

# %% Getting label_keys
label_keys = pd.read_csv(f'{path}/train.csv')
class_names = label_keys.loc[label_keys['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).unique()

# Counts of each label
label_totals = label_keys.loc[label_keys['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()

# Counts of number of label_keys per image
img_label_freq = label_keys.loc[label_keys['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().value_counts()

# %% Grabbing images with just one label
imgs_1label = label_keys.loc[label_keys['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')).value_counts()
imgs_1label = imgs_1label[imgs_1label == 1].index

# %%
imgs_label = np.array([label[1] for label in imgs_1label])  # the label_keys for the singlely labeled images
img_names = [label[0] for label in imgs_1label]             # the names of each singlely labeled images

# Names of images of each single label
single_img_names = []
for label in class_names:
    single_img = img_names[np.argwhere(imgs_label == label).flatten()[0]]
    single_img_names.append(single_img)

# Plotting the single images
fig, axs = plt.subplots(1, len(class_names), figsize=(7.5, 3), layout='constrained')
for label, img_name, ax in zip(class_names, single_img_names, axs.flatten()):
    # Getting img
    img = Image.open(f'{path}/train_images/{img_name}')
    ax.imshow(img)

    # Formatting
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{img_name}\n{label}')
