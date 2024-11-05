# -*- coding: utf-8 -*-
"""
Trying out pytorch to create a basic NN for classifying images.

This is largely based on: https://www.kaggle.com/code/artgor/segmentation-in-pytorch-using-convenient-tools
Other useful resources include: 
    * https://www.youtube.com/watch?v=V_xro1bcAuA


Created on Mon Nov  4 12:07:36 2024

@author: J. Monaco
"""
# %% Imports

# Analysis
import pandas as pd
import kaggle_helpers as kh
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset


# system tools
import os
import tqdm

# Plotting
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cm'

# %% Loading in the image data
kpath = './understanding_cloud_organization'     # path to images
label_keys = pd.read_csv('better_df.csv',
                         converters={'Fish': kh.str2intarr,
                                     'Flower': kh.str2intarr,
                                     'Gravel': kh.str2intarr,
                                     'Sugar': kh.str2intarr})       # Mask dataframe in better format

# Number of files
N_labeled = len(os.listdir(f'{kpath}/train_images'))
N_test = len(os.listdir(f'{kpath}/test_images'))

# Getting info about pictures by investigating a random image
rand_img_df = label_keys.sample(1).iloc[0]
rand_img = kh.get_img(rand_img_df.im_id, kpath)
ht, wd, _ = rand_img.shape  # height and width of the images
N_px = int(ht * wd)         # Number of pixels per image

# %% Separating out training and validation data
frac_training = 0.8                                 # Fraction of images to choose for training
num_training = int(N_labeled * frac_training)
training_keys = label_keys.sample(num_training)     # Random training data
valid_keys = label_keys.loc[~label_keys.index.isin(training_keys.index)]


# --- Setting up training data --- #
batch_sz = 32                                       # How many images to consider per batch
num_workers = 4
train_dataset = kh.CloudDataset(training_keys, datatype='train')
train_loader = DataLoader(train_dataset,
                          batch_size=batch_sz,
                          num_workers=num_workers)

# --- Setting up validation data --- #
valid_dataset = kh.CloudDataset(valid_keys, datatype='test')
valid_loader = DataLoader(valid_dataset,
                          batch_size=batch_sz,
                          num_workers=num_workers)

# %% Building model

# --- NN Structure --- #


class CloudClassr(nn.Module):
    def __init__(self, N, layer_size):
        super(CloudClassr, self).__init__()
        self.linear_ReLU_stack = nn.Sequential(
            nn.Linear(N, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, N // 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Whole flattened images as input
        x = x.flatten(start_dim=1, end_dim=-1)

        # Run through NN
        x_logits = self.linear_ReLU_stack(x)

        # turn logits -> pred probs -> pred labels
        x_pred = torch.round(torch.sigmoid(x_logits))
        return x_pred.squeeze()


# Device for data and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating an instance of the model on the target device
model = CloudClassr(3 * N_px, 8)
model.to(device)

# %% Training the model
criterion = nn.MSELoss()                                    # Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # Gradient optimizer
epochs = 32    # number of training loops                   # Number of training epochs


for epoch in tqdm.trange(epochs):

    # --- Training Model --- #
    model.train()   # Setting model to training mode

    for data, target in tqdm.tqdm(train_loader):
        # Forward pass: Predicting the labels on the training data
        X_train = torch.Tensor(data).float().to(device)
        X_pred = model(X_train)

        # Calculate loss for this batch
        X_truth = torch.Tensor(target).float().to(device)
        X_truth = X_truth.flatten(1, -1)
        loss = criterion(X_pred, X_truth)

        # Optimizer zero grad
        optimizer.zero_grad() 

        # Backpropagation
        loss.backward()

        # gradient descent)
        optimizer.step()
    model.eval()

    # --- Evaluating model
