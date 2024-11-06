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
import kaggle_helpers as kh
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# system tools
import os
import tqdm

# Plotting
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cm'

# %% Loading in the image data
print('Loading in the training data...')
kpath = './understanding_cloud_organization'     # path to images

# Mask dataframe in better format
label_keys = pd.read_csv('better_df.csv',
                         converters={'Fish': kh.str2intarr,
                                     'Flower': kh.str2intarr,
                                     'Gravel': kh.str2intarr,
                                     'Sugar': kh.str2intarr})

# Number of files
N_labeled = len(os.listdir(f'{kpath}/train_images'))
N_test = len(os.listdir(f'{kpath}/test_images'))

# Getting info about pictures by investigating a random image
rand_img_df = label_keys.sample(1).iloc[0]
rand_img = kh.get_img(rand_img_df.im_id, kpath)
ht, wd, _ = rand_img.shape  # height and width of the images
N_px = int(ht * wd)         # Number of pixels per image

# %% Separating out training and validation data
print('Creating training and validation datasets...')
frac_training = 0.8                                 # Fraction of images to choose for training
num_training = int(N_labeled * frac_training)
training_keys = label_keys.sample(num_training)     # Random training data
valid_keys = label_keys.loc[~label_keys.index.isin(training_keys.index)]


# --- Setting up training data --- #
batch_sz = 32                                        # How many images to consider per batch
train_dataset = kh.CloudDataset(training_keys, datatype='train')
train_loader = DataLoader(train_dataset,
                          batch_size=batch_sz,
                          shuffle=True)
print(f'Data divided in to {len(train_loader)} batches of {batch_sz} images each.')

# --- Setting up validation data --- #
valid_dataset = kh.CloudDataset(valid_keys, datatype='train')
valid_loader = DataLoader(valid_dataset,
                          batch_size=batch_sz)

# %% Building model
print('Creating NN model...')


class CloudClassr(nn.Module):
    '''
    NN structure for cloud classification. 

    Input layer should have 3x as many inputs as there are pixels, as there are 3 data channels 

    Output layer should have 1x as many outputs as there are pixels, as it's just a mask
    '''

    def __init__(self, N_px, layer_size):
        super(CloudClassr, self).__init__()
        self.linear_ReLU_stack = nn.Sequential(
            nn.Linear(in_features=3 * N_px, out_features=layer_size),
            nn.ReLU(),
            nn.Linear(in_features=layer_size, out_features=layer_size),
            nn.ReLU(),
            nn.Linear(in_features=layer_size, out_features=layer_size),
            nn.ReLU(),
            nn.Linear(in_features=layer_size, out_features=N_px),
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
model = CloudClassr(N_px, 8)
model.to(device)

# %% Training the model
# criterion = nn.CrossEntropyLoss()                             # Loss function for multiclass data
criterion = nn.BCELoss()                                        # Loss function for binary class data
optimizer = torch.optim.Adam(model.parameters(), lr=1e0)   # Gradient optimizer
# optimizer = torch.optim.SGD(params=model.parameters(),
#                             lr=0.1)
epochs = 5                                                 # Number of training epochs

print(f'Training NN with {epochs} epochs, each with {batch_sz} images...')

for epoch in tqdm.trange(epochs, desc='Epochs: '):

    # --- Training Model --- #
    train_loss = 0
    data_iter = tqdm.tqdm(train_loader, desc='    Train. Batch: ',
                          postfix={"Training Loss": 0})
    for data, target in data_iter:
        model.train()   # Setting model to training mode

        # Forward pass: Predicting the labels on the training data
        X_train = torch.Tensor(data).float().to(device)
        X_pred = model(X_train)

        # Calculate loss for this batch
        X_truth = torch.Tensor(target).float().to(device)
        X_truth = X_truth.flatten(1, -1)
        loss = criterion(X_pred, X_truth)
        train_loss += loss

        # Optimizer zero grad
        optimizer.zero_grad() 

        # Backpropagation
        loss.backward()

        # gradient descent)
        optimizer.step()
        print(f'Training loss: {loss:.3f}')
        # data_iter.set_postfix({"Training Loss": loss.item()})
    train_loss /= len(train_loader)

# %% Evaluating model
model.eval()
test_loss, test_acc = 0, 0
with torch.inference_mode():
    data_iter = tqdm.tqdm(valid_loader, desc='    Valid. Batch: ',
                          postfix={"Pct. Accuracy": 0})
    for data, target in data_iter:
        # Forward pass
        X_test = torch.Tensor(data).float().to(device)
        test_pred = model(X_test)

        # Calculate loss (accumulatively)
        test_truth = torch.Tensor(target).float().to(device)
        test_truth = test_truth.flatten(1, -1)
        test_loss += criterion(test_pred, test_truth)

        # Calculate accuracy
        correct = torch.eq(test_truth, test_pred).sum().item()
        acc = (correct / len(test_pred) / N_px) * 100
        test_acc += acc
        data_iter.set_postfix({"Pct. Accuracy": acc})

    # Calculate the test loss average per batch
    test_loss /= len(valid_loader)

    # Calculate the test acc average per batch
    test_acc /= len(valid_loader)
print(f'For epoch {epoch}, there was an average training loss per batch of \
{train_loss:.2f}, average test loss of {test_loss:.2f}, and accuracy of \
{test_acc:.1f}%')
