# -*- coding: utf-8 -*-
"""
Trying out pytorch to create a basic NN for classifying images.

This is largely based on the following resources:
    * https://www.kaggle.com/code/artgor/segmentation-in-pytorch-using-convenient-tools
    * https://github.com/mrdbourke/pytorch-deep-learning/blob/main/video_notebooks/03_pytorch_computer_vision_video.ipynb

Other useful resources include: 
    * https://www.youtube.com/watch?v=V_xro1bcAuA


Created on Mon Nov  4 12:07:36 2024

@author: J. Monaco
"""
# %% Imports

# Analysis
import pandas as pd
import numpy as np
import kaggle_helpers as kh
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# system tools
import os
import tqdm

# Plotting
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cm'
import pickle

# %% Loading in the image data
print('Loading in the training data...')
kpath = './understanding_cloud_organization'     # path to images
with open('better_df.pkl', 'rb') as f:
    label_keys = pickle.load(f)

# Number of files
N_labeled = len(os.listdir(f'{kpath}/train_images'))
N_test = len(os.listdir(f'{kpath}/test_images'))

# Getting info about pictures by investigating a random image
rand_img_df = label_keys.sample(1).iloc[0]
rand_img = kh.get_img(rand_img_df.im_id, kpath)
ht, wd, n_clrs = rand_img.shape  # height and width of the images

# %% Separating out training and validation data
print('Creating training and validation datasets...')
frac_training = 0.85                                 # Fraction of images to choose for training
num_training = int(N_labeled * frac_training)
training_keys = label_keys.sample(num_training)     # Random training data
valid_keys = label_keys.loc[~label_keys.index.isin(training_keys.index)]
print(f'Using {frac_training * 100:.0f}% of data for training...')

# --- Setting up training data --- #
batch_sz = 8                                        # How many images to consider per batch
train_dataset = kh.CloudDataset_PCA(training_keys,
                                    img_paths=f'{kpath}/train_images')
train_loader = DataLoader(train_dataset,
                          batch_size=batch_sz,
                          shuffle=True)
print(f'Training data divided in to {len(train_loader)} batches of {batch_sz} images each.')

# --- Setting up validation data --- #
valid_dataset = kh.CloudDataset_PCA(valid_keys,
                                    img_paths=f'{kpath}/train_images')
valid_loader = DataLoader(valid_dataset,
                          batch_size=batch_sz)

# %% Building model
print('Creating NN model...')


class CloudClassr(nn.Module):
    '''
    Takes in 1 channel PCA'd image, and outputs masks for all four classes. 
    '''

    def __init__(self):
        super(CloudClassr, self).__init__()

        # Encoder: Downsampling with convolutions and max-pooling
        self.enc1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Decoder: Upsampling with transposed convolutions
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, 4, kernel_size=1)  # Output layer for binary mask

    def forward(self, x):
        # Encoding (Downsampling)
        x1 = F.relu(self.enc1(x))
        x2 = F.max_pool2d(x1, 2)  # Down to 700x1050
        x2 = F.relu(self.enc2(x2))
        x3 = F.max_pool2d(x2, 2)  # Down to 350x525
        x3 = F.relu(self.enc3(x3))

        # Decoding (Upsampling)
        x = self.up1(x3)          # Up to 700x1050
        x = F.relu(x + x2)        # Skip connection from encoder
        x = self.up2(x)           # Up to 1400x2100
        x = F.relu(x + x1)        # Skip connection from encoder

        # Final 1x1 convolution to get binary segmentation output
        x = torch.sigmoid(self.final_conv(x))
        return x.squeeze()


# Device for data and model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating an instance of the model on the target device
model = CloudClassr()
model.to(device)

# %% Training the model

# --- Loss functions and gradient descent optimizer --- #
# criterion = nn.CrossEntropyLoss()                         # Loss function for multiclass data
criterion = nn.BCELoss()                                    # Loss function for binary class data
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # Gradient optimizer
# optimizer = torch.optim.SGD(params=model.parameters(),
#                             lr=0.1)

# --- Training --- #
epochs = 12              # Number of training epochs
print(f'Training NN with {epochs} epochs...')

# Losses and accuracies to plot for each epoch
train_losses = np.ones(epochs) * np.nan
test_losses = np.ones(epochs) * np.nan
test_accs = np.ones(epochs) * np.nan
test_DICEs = np.ones(epochs) * np.nan

# Training / evaluation loop
for epoch in tqdm.trange(epochs, desc='Epochs: '):

    # --- Training Loop --- #
    train_loss = 0
    data_iter = tqdm.tqdm(train_loader, desc='    Train. Batch: ',
                          postfix={"Training Loss": 0})
    for data, target in data_iter:
        model.train()   # Setting model to training mode

        # Forward pass: Predicting the labels on the training data
        X_train = data
        X_pred = model(X_train)

        # Calculate loss for this batch
        X_truth = target
        loss = criterion(X_pred, X_truth)
        train_loss += loss
        data_iter.set_postfix({"Training Loss": loss.item()})

        # Optimizer zero grad
        optimizer.zero_grad() 

        # Backpropagation
        loss.backward()

        # gradient descent)
        optimizer.step()

    train_loss /= len(train_loader)
    train_losses[epoch] = train_loss

    # --- Evaluation Loop --- #
    epoch_loss, epoch_acc, epoch_DICE = 0, 0, 0
    with torch.inference_mode():
        data_iter = tqdm.tqdm(valid_loader, desc='    Valid. Batch: ',
                              postfix={"Pct. Accuracy": 0})
        for data, target in data_iter:
            model.eval()    # set model to evaluation mode

            # Forward pass
            X_test = torch.Tensor(data).float().permute(0, 3, 1, 2).to(device)
            test_pred = model(X_test)

            # Calculate loss (accumulatively)
            test_truth = torch.Tensor(target).float().to(device)
            epoch_loss += criterion(test_pred, test_truth)

            # Calculate accuracy
            num_correct = torch.eq(test_truth, test_pred.round()).sum().item()
            batch_acc = (num_correct / test_pred.numel()) * 100
            epoch_acc += batch_acc
            data_iter.set_postfix({"Pct. Accuracy": batch_acc})

            # Calculate DICE score
            batch_DICE = 2 * num_correct / (test_pred.numel() + test_truth.numel())
            epoch_DICE += batch_DICE

        # Calculate the average test loss for this epoch
        epoch_loss /= len(valid_loader)
        test_losses[epoch] = epoch_loss

        # Calculate the average test acc for this epoch
        epoch_acc /= len(valid_loader)
        test_accs[epoch] = epoch_acc

        # Calc avg DICE score for this epoch
        epoch_DICE /= len(valid_loader)
        test_DICEs[epoch] = epoch_DICE

    print(f'For epoch {epoch}, there was an average training loss per batch of \
    {epoch_loss:.2f}, average test loss of {epoch_loss:.2f}, and accuracy of \
    {epoch_acc:.1f}%')

# %% --- Saving the model --- #
torch.save(obj=model.state_dict(),
           f='cloudClassr_allclass_v1')

# %% Plotting metrics over the course of training
fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')

# Plotting losses
plt_epochs = np.arange(epochs)
ax = axs[0]
ax.plot(plt_epochs, train_losses, '.-', label='Training Loss')
ax.plot(plt_epochs, test_losses, '.-', label='Testing Loss')
ax.legend()
ax.set_xlabel('Epoch Number')
ax.set_ylabel('Loss')
ax.set_title('Evolution of Losses over Training')

# Plotting DICE score
ax_DICE = axs[1]
ax_DICE.plot(plt_epochs, test_DICEs, '.-')
ax_DICE.set_ylabel('DICE Score')
ax_DICE.set_xlabel('Epoch Number')
ax_DICE.set_title('Evaluation Metrics over Training')

# Plotting accuracy
ax_acc = ax_DICE.twinx()
ax_acc.plot(plt_epochs, test_accs, 'r.-')
ax_acc.set_ylabel(r'Model Accuracy [\%]', color='r')
ax_acc.tick_params(axis='y', color='r')

# Formatting
plt.show()
