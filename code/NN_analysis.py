# -*- coding: utf-8 -*-
"""
Looking at the results from different trained classifiers 

Created on Mon Nov 11 09:29:36 2024

@author: J. Monaco
"""
# %% Imports

# Plotting
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cm'
import seaborn as sns
import matplotlib.gridspec as GS
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# system tools
import tqdm
import pickle

# Analysis
import numpy as np
import kaggle_helpers as kh
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# %% Loading in the data
print('Loading in the training data...')
torch.manual_seed(0)    # Pytorch seed

# Opening the better df pkl located in the same directory as this file
with open('better_df.pkl', 'rb') as f:
    label_keys = pickle.load(f)
N_samples = int(1e3)                                    # Number of images to do analysis on
label_keys = label_keys.sample(N_samples, random_state=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for data and model
kpath = './understanding_cloud_organization'            # kaggle data path, containing the training images
downscale_factor = 4                                    # Approximate factor of decimation
batch_sz = 32                                           # How many images to consider per batch
train_dataset = kh.CloudDataset_PCA_scaled(label_keys,
                                           downscale_factor=downscale_factor,
                                           img_paths=f'{kpath}/train_images',
                                           device=device)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_sz)

# %% Building model
print('Loading NN model...')

# Creating an instance of the model on the target device
model_path = './model_scripted.pt'
model = torch.jit.load(model_path)
model.to(device)

# %% evaluating the model
thresholds = [0.14, 0.13, 0.14, 0.12]  # thresholds for raw logits, found by iterating over and selected highest avg DICE

# --- Loss functions and gradient descent optimizer --- #
criterion = nn.BCELoss()                                    # Loss function for binary class data
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # Gradient optimizer

# --- Evaluation Loop --- #
epoch_loss, epoch_acc, epoch_DICE = 0, 0, 0
with torch.inference_mode():
    data_iter = tqdm.tqdm(train_loader, desc='    Valid. Batch: ',
                          postfix={"DICE": 0})
    for data, target in data_iter:
        # Forward pass
        X_test = data
        test_pred = model(X_test)

        # Calculate loss (accumulatively)
        test_truth = target
        epoch_loss += criterion(test_pred, test_truth)

        # Getting the logits
        pred_np = test_pred.cpu().numpy()           # Convert raw logits to numpy array

        # Convert logits to mask values using thresholds
        pred_np_converted = np.copy(pred_np)
        for classnum, threshold in enumerate(thresholds):
            pred_np_converted[:, classnum, :, :] = np.where(pred_np_converted[:, classnum, :, :] > threshold, 1, 0)
        test_truth = test_truth.cpu().numpy()       # Convert truth to numpy array

        # Calculate DICE score
        batch_DICE = kh.dice(test_truth,
                             pred_np_converted)
        data_iter.set_postfix({"DICE": batch_DICE})
        epoch_DICE += batch_DICE

    # Calculate the average test loss for this epoch
    epoch_loss /= len(train_loader)

    # Calc avg DICE score for this epoch
    epoch_DICE /= len(train_loader)

    print(f'There was an average training loss per batch of \
    {epoch_loss:.2f}, average test loss of {epoch_loss:.2f}, and DICE of \
    {epoch_DICE:.2f}.')

# %% Checking outputs of model for last batch ran as a gut check
fig, axs = plt.subplots(3, 4, figsize=(12, 4.5), layout='constrained',
                        sharey='row')

for label_num, label in enumerate(train_dataset.labels):
    axs[0, label_num].set_title(label)  # Setting title

    # --- Histogram of raw logits --- #
    logits_1label = pred_np[:, label_num, :, :].flatten()
    sns.histplot(logits_1label, ax=axs[0, label_num], stat='density')
    axs[0, label_num].set_yscale('log')
    axs[0, label_num].set_yticks([])
    axs[0, label_num].set_ylabel('')

    # --- Histogram of predicted masks --- #
    pred_masks = pred_np_converted[:, label_num, :, :].flatten()
    sns.histplot(pred_masks, ax=axs[1, label_num], stat='density',
                 binwidth=0.02)
    axs[1, label_num].set_yticks([])
    axs[1, label_num].set_ylabel('')

    # --- Histogram of truth masks --- #
    truth_masks = test_truth[:, label_num, :, :].flatten()
    sns.histplot(truth_masks, ax=axs[2, label_num], stat='density',
                 binwidth=0.02)
    axs[2, label_num].set_yticks([])
    axs[2, label_num].set_ylabel('')

# Formatting plots
axs[0, 0].set_ylabel('Raw Logits')
axs[1, 0].set_ylabel('Predicted Masks')
axs[2, 0].set_ylabel('Truth Masks')
fig.suptitle(f'Distribution of Masks and Logits for A Single Batch\nDICE={batch_DICE:.2f}')
plt.savefig('../figs/logit_hist.pdf', dpi=400, bbox_inches='tight')
plt.show()

# %% Looking at logits and masks

# --- Getting an image --- #
img_num = np.random.randint(0, data.shape[0])
img_PCA = X_test.cpu().numpy()[img_num, 0, :, :]
img_mask = test_truth[img_num, :, :, :]
img_pred = pred_np_converted[img_num, :, :, :]
img_logits = pred_np[img_num, :, :, :]

# --- Setting up axes --- #
fig, axs = plt.subplots(2, 6, figsize=(12, 3.5), layout='constrained')
gs = axs[0, 3].get_gridspec()
for ax in axs[:, 2:4].ravel():
    ax.remove()
ax_logit = axs[:, 0:2]
ax_PCA = axs[:, 4:]
class_names = ['Fish', 'Flower', 'Gravel', 'Sugar']
colors = ['c', 'm', 'y', 'g']

# --- Plotting the image --- #
ax = fig.add_subplot(gs[:, 2:4])
ax.imshow(img_PCA, cmap='gray')
# Plotting masks
for class_num, (class_name, color) in enumerate(zip(class_names, colors)):
    mask_img = img_mask[class_num, :, :]
    ax.contour(mask_img, colors=color, linewidths=0.3)
    cmap = ListedColormap(['w', color])
    ax.imshow(mask_img, alpha=mask_img * 0.2, cmap=cmap)
# Formatting
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('PCA and Truth Masks')

# --- Plotting the logits --- #
for class_num, (ax, class_name, color) in enumerate(zip(ax_logit.ravel(), class_names, colors)):
    # Plotting logits
    pred = img_logits[class_num, :, :]
    im = ax.imshow(pred)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='horizontal', shrink=0.1)

    # Plotting predicted masks
    mask_img = img_pred[class_num, :, :]
    ax.contour(mask_img, colors='r', linewidths=0.05, alpha=0.5)
    cmap = ListedColormap(['w', 'r'])
    ax.imshow(mask_img, alpha=mask_img * 0.2, cmap=cmap)

    # Plotting truth masks
    mask_img = img_mask[class_num, :, :]
    ax.contour(mask_img, colors=color, linewidths=0.5)
    cmap = ListedColormap(['w', color])
    ax.imshow(mask_img, alpha=mask_img * 0.2, cmap=cmap)

    # Formatting
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{class_name} Logits and Mask')


# --- Masks on real image --- #
for class_num, (ax, class_name, color) in enumerate(zip(ax_PCA.ravel(), class_names, colors)):
    # Plotting logits
    im = ax.imshow(img_PCA, cmap='gray')

    # Plotting predicted masks
    mask_img = img_pred[class_num, :, :]
    ax.contour(mask_img, colors='r', linewidths=0.1, alpha=0.5)
    cmap = ListedColormap(['w', 'r'])
    ax.imshow(mask_img, alpha=mask_img * 0.2, cmap=cmap)

    # Plotting truth masks
    mask_img = img_mask[class_num, :, :]
    ax.contour(mask_img, colors=color, linewidths=0.5)
    cmap = ListedColormap(['w', color])
    ax.imshow(mask_img, alpha=mask_img * 0.15, cmap=cmap)

    # Formatting
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{class_name} Mask on PCA')
plt.savefig('../figs/mask_example.pdf', bbox_inches='tight', dpi=400)
plt.show()
