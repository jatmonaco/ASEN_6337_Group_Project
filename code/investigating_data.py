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
import kaggle_helpers as KH
from matplotlib.pyplot import savefig
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cm'

# %% Getting data
path = './understanding_cloud_organization'
print(f'Files and Folders available: {os.listdir(path)}')

# Number of files
N_labeled = len(os.listdir(f'{path}/train_images'))
N_test = len(os.listdir(f'{path}/test_images'))
print(f'{N_labeled} training images and {N_test} test images available')

# %% Getting label_keys
label_keys = pd.read_csv(f'{path}/train.csv')
label_keys['label'] = label_keys['Image_Label'].apply(lambda x: x.split('_')[1])
label_keys['im_id'] = label_keys['Image_Label'].apply(lambda x: x.split('_')[0])
class_names = label_keys['label'].unique()   # Name of all labels

# Counts of each label
class_totals = label_keys.loc[label_keys['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()
print('Class occurences: ', class_totals, '\n')

# Counts of number of label_keys per image
img_label_freq = label_keys.loc[label_keys['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().value_counts()
print('Class occurences within images: ', img_label_freq)

# %% Putting the training data in better pandas format
'''
Each picutre is only represented by 1 row, with columns: 
        * 'im_id': the name of the image 
        * 'Flower', 'Gravel', 'Sugar', 'Fish': The rle label for that row's image
        * num_labels: the number of valid labels 
'''
training_cols = ['im_id', *class_names, 'num_labels']
training_df = pd.DataFrame(columns=training_cols)
training_df['im_id'] = label_keys['im_id'].unique()
for idx, row in training_df.iterrows():
    img_df = label_keys[label_keys.im_id == row.im_id]
    row['num_labels'] = 0
    for class_name in class_names:
        rle = img_df[img_df.label == class_name].EncodedPixels.dropna().to_numpy()
        if rle.size < 1:
            continue
        rle_str = rle[0]
        rle = np.fromstring(rle_str, sep=' ', dtype=int)
        row[f'{class_name}'] = rle
        row['num_labels'] += 1
    training_df.iloc[idx] = row

# %% Saving this as a .csv
write_csv = training_df.copy()


def arr2str(arr):
    if not np.isnan(arr).any():
        string = " ".join(map(str, arr))
    else:
        string = np.nan
    return string


for label in class_names:
    write_csv[label] = write_csv[label].apply(lambda x: arr2str(x))
write_csv.to_csv('better_df.csv', index=False)
# %%  Plotting single images
single_label_imgs = training_df[training_df.num_labels == 1]

fig, axs = plt.subplots(1, len(class_names), figsize=(7.5, 2), layout='constrained')
colors = ['c', 'm', 'y', 'g']
for class_name, ax, color in zip(class_names, axs.flatten(), colors):
    # Getting img
    single_label_img = single_label_imgs[~single_label_imgs[f'{class_name}'].isna()].iloc[0]
    img_name = single_label_img.im_id
    img = Image.open(f'{path}/train_images/{img_name}')
    ax.imshow(img)

    # Getting mask
    mask_rle = single_label_img[f'{class_name}']
    mask_img = KH.rle_decode(mask_rle)
    ax.contour(mask_img, colors=color)
    ax.imshow(mask_img, alpha=mask_img * 0.5, cmap='gray')

    # Formatting
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{img_name}\n{class_name}')
fig.suptitle('Singly-Labeled Images and Their Masks')
savefig('../figs/single_labels.pdf', bbox_inches='tight', dpi=400)

# %%  Plotting image with all four labels


fig, ax = plt.subplots(1, 1, figsize=(3, 3), layout='constrained')

all_label_img = training_df[training_df.num_labels == len(class_names)].iloc[0]
img_name = all_label_img.im_id
img = Image.open(f'{path}/train_images/{img_name}')
ax.imshow(img)

# Getting mask
colors = ['c', 'm', 'y', 'g']
for class_name, color in zip(class_names, colors):
    mask_rle = all_label_img[f'{class_name}']
    mask_img = KH.rle_decode(mask_rle)
    ax.contour(mask_img, colors=color)
    ax.imshow(mask_img, alpha=mask_img * 0.25, cmap='gray')

# Formatting
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f'Image with All Classes\n{img_name}')
savefig('../figs/all_labels.pdf', bbox_inches='tight', dpi=400)
