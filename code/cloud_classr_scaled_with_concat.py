# -*- coding: utf-8 -*-
"""
Downsampling images and training NN. Based on cloud_classr.py

Created on Mon Nov  4 12:07:36 2024

@author: J. Monaco
"""

# TODO: use on small images, mask downscale is 4 image downscale is 1
# %% Imports

# Analysis
import numpy as np
import kaggle_helpers as kh
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

# system tools
import tqdm
import pickle

# Plotting
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cm'
import seaborn as sns
from matplotlib.pyplot import savefig

descriptor = '_concat'
# %% Loading in the image data
print('Loading in the training data...')

# Opening the better df pkl located in the same directory as this file
with open('better_df.pkl', 'rb') as f:
    label_keys = pickle.load(f)

# Number of files
kpath = './understanding_cloud_organization'            # kaggle data path, containing the training images
# label_keys = label_keys.sample(1000)                    # just using 1000 images for testing things
N_labeled = len(label_keys)

# Getting info about pictures by investigating a random image
rand_img_df = label_keys.sample(1).iloc[0]
rand_img = kh.get_img(rand_img_df.im_id, kpath)
ht, wd, n_clrs = rand_img.shape  # dimentionality of the images

# %% Separating out training and validation data
print('Creating training and validation datasets...')

# --- Setting seeds --- #
torch.manual_seed(0)    # Pytorch seeds

frac_training = 0.9                                 # Fraction of images to choose for training
num_training = int(N_labeled * frac_training)       # Number of images for training
training_keys = label_keys.sample(num_training, random_state=0)     # Random images for training
valid_keys = label_keys.loc[~label_keys.index.isin(training_keys.index)]    # Rest of the images for validation
print(f'Using {frac_training * 100:.0f}% of data for training...')

# --- Setting up training data --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for data and model
batch_sz = 32                                      # How many images to consider per batch
downscale_factor = 4                                # Approximate factor of decimation

train_dataset = kh.CloudDataset_PCA_scaled(training_keys,
                                           downscale_factor=downscale_factor,
                                           img_paths=f'{kpath}/train_images',
                                           device=device)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_sz,
                          shuffle=True)
print(f'Training data divided in to {len(train_loader)} batches of {batch_sz} images each.')

# --- Setting up validation data --- #
valid_dataset = kh.CloudDataset_PCA_scaled(valid_keys,
                                           downscale_factor=downscale_factor,
                                           img_paths=f'{kpath}/train_images',
                                           device=device)
valid_loader = DataLoader(valid_dataset,
                          batch_size=batch_sz)

# %% Defining focal loss

# One-hot encoding function (modified)


def one_hot(index, classes):
    size = index.size()  # + (classes,)
    view = index.size()  # + (1,)

    # Ensure target is of type int64 for scatter operation
    index = index.to(torch.int64)

    mask = torch.zeros(*size, device=index.device)
    index = index.view(*view)

    return mask.scatter_(1, index, 1.0)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.shape[-1])
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = loss * (1 - logit) ** self.gamma  # focal loss

        return loss.sum()

# chat gpt wrote this


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Apply sigmoid to the predictions to get them in the range [0, 1]
        y_pred = torch.sigmoid(y_pred)

        # Calculate the intersection and union over the spatial dimensions (dim 2 and 3)
        intersection = (y_pred * y_true).sum(dim=[2, 3])
        union = y_pred.sum(dim=[2, 3]) + y_true.sum(dim=[2, 3]) - intersection

        # Calculate IoU per channel and then average over channels
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1 - iou.mean(dim=1)  # Mean over channels
        return iou_loss.mean()  # Mean over batch


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Apply sigmoid to the predictions to get them in the range [0, 1]
        y_pred = torch.sigmoid(y_pred)

        # Calculate the intersection and sums over the spatial dimensions (dim 2 and 3)
        intersection = (y_pred * y_true).sum(dim=[2, 3])
        sum_pred = y_pred.sum(dim=[2, 3])
        sum_true = y_true.sum(dim=[2, 3])

        # Calculate DICE per channel and then average over channels
        dice = (2 * intersection + self.smooth) / (sum_pred + sum_true + self.smooth)
        dice_loss = 1 - dice.mean(dim=1)  # Mean over channels
        return dice_loss.mean()  # Mean over batch


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
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # Increase input channels due to concatenation
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # Increase input channels due to concatenation
        self.final_conv = nn.Conv2d(16, 4, kernel_size=1)  # Output layer for binary mask

        self.conv_dec1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_dec2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoding (Downsampling)
        x1 = F.relu(self.enc1(x))
        x2 = F.max_pool2d(x1, 2)  # Down by factor of 2
        x2 = F.relu(self.enc2(x2))
        x3 = F.max_pool2d(x2, 2)  # Down by factor of 2
        x3 = F.relu(self.enc3(x3))

        # Decoding (Upsampling)
        x = self.up1(x3)  # Concatenate along channels (dim=1)
        x = F.relu(self.conv_dec1(torch.cat([x, x2], dim=1)))  # now 32 wide
        x = self.up2(x)   # Concatenate along channels (dim=1)
        x = F.relu(self.conv_dec2(torch.cat([x, x1], dim=1)))  # now 16

        # Final 1x1 convolution to get binary segmentation output
        x = torch.sigmoid(self.final_conv(x))
        return x.squeeze()


# Creating an instance of the model on the target device
model = CloudClassr()
model.to(device)


# %% Training the model
print("Training NN Model...")
# --- Thresholds to calculate DICE --- #
thresholds = [0.14, 0.13, 0.14, 0.12]  # thresholds for raw logits, found by iterating over and selected highest avg DICE

# --- Loss functions and gradient descent optimizer --- #
criterion = IoULoss()                                       # Loss function for binary class data
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # Gradient optimizer

# --- Training --- #
epochs = 20            # Number of training epochs
print(f'Training NN with {epochs} epochs...')

# Losses and accuracies to plot for each epoch
train_losses = np.ones(epochs) * np.nan
test_losses = np.ones(epochs) * np.nan
test_DICEs = np.ones(epochs) * np.nan

# Training / evaluation loop
for epoch in tqdm.trange(epochs, desc='Epochs: '):

    # --- Training Loop --- #
    train_loss = 0
    data_iter = tqdm.tqdm(train_loader, desc='    Train. Batch: ',
                          postfix={"Training Loss": 0}, leave=True)
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
                              postfix={"DICE": 0})
        for data, target in data_iter:
            model.eval()    # set model to evaluation mode

            # Forward pass
            X_test = data
            test_pred = model(X_test)

            # Calculate loss (accumulatively)
            test_truth = target

            epoch_loss += criterion(test_pred, test_truth)  # using the focal loss function means need to do binary loss

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
        epoch_loss /= len(valid_loader)
        test_losses[epoch] = epoch_loss

        # Calc avg DICE score for this epoch
        epoch_DICE /= len(valid_loader)
        test_DICEs[epoch] = epoch_DICE

    print(f'For epoch {epoch}, there was an average training loss per batch of \
    {epoch_loss:.2f}, average test loss of {epoch_loss:.2f}, and DICE of \
    {epoch_DICE:.2f}.')

# %% --- Saving the model --- #
model_scripted = torch.jit.script(model)  # Export to TorchScript
model_scripted.save(f'model{descriptor}.pt')  # Save

# %% Plotting metrics over the course of training

import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
fig, ax = plt.subplots(1, 1, figsize=(8, 4), layout='constrained')

# Plotting losses
plt_epochs = np.arange(epochs)
ax.plot(plt_epochs, train_losses, '.-', label='Training Loss')
ax.plot(plt_epochs, test_losses, '.-', label='Testing Loss')
ax.legend()
ax.set_xlabel('Epoch Number')
ax.set_ylabel('Loss')
ax.set_title('Evolution of Losses over Training')

# Plotting DICE score
ax_DICE = ax.twinx()
ax_DICE.plot(plt_epochs, test_DICEs, 'r.--')
ax_DICE.set_ylabel('DICE Score', color='r')
ax_DICE.tick_params(axis='y', color='r')

# Formatting
plt.show()
savefig(f'../figs/training_progression{descriptor}.pdf', bbox_inches='tight', dpi=600)
