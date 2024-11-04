# -*- coding: utf-8 -*-
"""
Translating HW3 to using pytorch. 
Exploring and displaying the HMI and truth data.

See: 
    * https://www.youtube.com/watch?v=ORMx45xqWkA
    * https://www.youtube.com/watch?v=V_xro1bcAuA

Created on Mon Oct 14 10:28:09 2024

@author: J. Monaco
"""

# %% Imports

# Analysis
import numpy as np
import sklearn.metrics as stats
from sklearn.neural_network import MLPClassifier as NN
import time
import kaggle_helpers as kh

# system tools
import os

# Plotting
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cm'
import seaborn as sns

# %% Reading the truth data
data_dir = './hw3_data'       # Parent directory that holds all data
X_truth = np.loadtxt(f'{data_dir}/Labeled_SDO_Data_for_Hw3_2024.csv',
                     delimiter=',', skiprows=0)
X_truth = X_truth[1:]                               # Discard the header

# Getting inputs
N = X_truth.size
K = len(set(X_truth))

# Putting truth into an image format
truth_img = np.reshape(X_truth, (int(N**0.5), -1))

# %% Plotting truth
fig, ax = plt.subplots(figsize=(3, 3.5), layout='constrained')
ax.imshow(truth_img, vmin=0, vmax=K - 1)
# Formatting the subplot
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
fig.suptitle(f'Truth Classifications\nK={K:.0f}')
plt.show()

# %% Loading in the files
scale = 255     # The max and min of the re-scaled and normalized images

# Finding all channel directories
channel_imgs = []
X_data = []
for data_file in os.listdir(data_dir):
    if data_file.startswith('AIA') and data_file.endswith('.npz'):
        print(f'Found data file {data_file}')
        data = np.load(data_dir + f'/{data_file}')      # Load data
        data = data[data.files[0]]
        data = kh.norm_matrix(data) * scale             # Norm and scale the data
        channel_imgs.append(data.T)                     # Save the data
        X = np.reshape(data.T, (N))
        X_data.append(X)
D = len(channel_imgs)                                   # Number of channels
channel_imgs = np.array(channel_imgs)                   # Collection of images
X_data = np.array(X_data)

# %% Plotting observational data

fig, ax = plt.subplots(figsize=(3, 3.5), layout='constrained')
img = kh.norm_matrix(channel_imgs[0:3, :, :])
img = np.einsum('kij->ijk', np.float64(img))
ax.imshow(img)

# Formatting the subplot
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
fig.suptitle('False Coloration of Data\nUsing First 3 Channels')
plt.show()

# %% Divding into training and validation datasets

# Determining training data indices
frac_training_max = 0.8                 # Maximum fraction of data to be used for training
frac_training_min = 0.1                # Minimum fraction of data to be used for training
num_trainings = 4
training_range = np.array([frac_training_min, frac_training_max]) * N
# Making sure to take whole rows for plotting purposes
training_range = (training_range - training_range % N**0.5).astype(int)

# Validation data and truth
X_vdata = X_data[:, training_range[-1]:]
X_vtruth = X_truth[training_range[-1]:]

# Different amounts of training data
training_idx = np.linspace(training_range[0], training_range[-1], num_trainings)
training_idx = (training_idx - training_idx % N**0.5).astype(int)

# Classifying with different amounts of training data
kappa_v = np.ones(num_trainings) * np.nan
pe_v = np.ones(num_trainings) * np.nan
po_v = np.ones(num_trainings) * np.nan
kappa_t = np.ones(num_trainings) * np.nan
pe_t = np.ones(num_trainings) * np.nan
po_t = np.ones(num_trainings) * np.nan


def fitNstats(classr, X_data, X_truth):
    '''
    Using a trained classifier, fit and calculate statistics on some data with 
    known truth values. 
    '''
    # Classify data
    X_pred = classr.predict(X_data.T)
    N = len(X_pred)

    # Compute stats
    R = stats.confusion_matrix(X_pred, X_truth)
    po = np.diag(R).sum() / N
    pe = np.sum(R, axis=0) @ np.sum(R, axis=1) / N**2
    kappa = stats.cohen_kappa_score(X_truth, X_pred)
    return X_pred, kappa, po, pe, R


# %% NN Implementation using scikitlearn


# NN parameters
hidden_layers = 2
layer_size = 8
hidden_layer_sizes = np.ones(hidden_layers, dtype=int) * layer_size
act_func = 'ReLU'
nn = NN(hidden_layer_sizes=hidden_layer_sizes)

# Classifying with different amounts of training data
num_trainings = len(training_idx)
kappa_v = np.ones(num_trainings) * np.nan
pe_v = np.ones(num_trainings) * np.nan
po_v = np.ones(num_trainings) * np.nan
kappa_t = np.ones(num_trainings) * np.nan
pe_t = np.ones(num_trainings) * np.nan
po_t = np.ones(num_trainings) * np.nan

# Timing each training
times = np.ones(num_trainings) * np.nan

for run, idx in enumerate(training_idx):
    # Training data and truth
    X_tdata = X_data[:, 0:idx]
    X_ttruth = X_truth[0:idx]

    # Training NN
    start = time.time()
    nn.fit(X_tdata.T, X_ttruth)        # Fitting
    end = time.time()
    print(f'Run {run:.0f} took {(end - start) / 60:.1f} mins to complete.')

    # Classification on the training data set
    X_tpred, kappa_t[run], po_t[run], pe_t[run], R = fitNstats(nn, X_tdata, X_ttruth)

    # Classification on the validation data set
    X_vpred, kappa_v[run], po_v[run], pe_v[run], R = fitNstats(nn, X_vdata, X_vtruth)

# %% Plotting the results of skl NN
fig, ax = plt.subplots(1, 3, figsize=(7.5, 3),
                       layout='constrained')
# x-axis is the percent of data used
pct_data = training_idx / N * 100

# Plotting statistics for the training set
ax[0].plot(pct_data, kappa_t, '.-', label=r'$\kappa$')
ax[0].plot(pct_data, po_t, '.--', label=r'$p_o$')
ax[0].plot(pct_data, pe_t, '.:', label=r'$p_e$')
ax[0].set_title('Training Data')
ax[0].set_xlabel('Amount of Image Trained On [%]')
ax[0].set_ylabel('Statistic Amplitude')
# Plotting statistics for the training set
ax[1].plot(pct_data, kappa_v, '.-', label=r'$\kappa$')
ax[1].plot(pct_data, po_v, '.--', label=r'$p_o$')
ax[1].plot(pct_data, pe_v, '.:', label=r'$p_e$')
ax[1].set_title('Validation Data')
ax[1].set_xlabel('Amount of Image Trained On [%]')
ax[1].sharey(ax[0])
ax[1].legend()

# Confusion Matrix
sns.heatmap(R, annot=True, cbar=False, square=True, ax=ax[2],
            annot_kws={'size': 6}, cmap=sns.cm.rocket_r)
ax[2].set_title('Confusion Matrix,\nMax Training')
ax[2].set_ylabel('Truth Cat.')
ax[2].set_xlabel('Predicted Cat.')

# Formatting and saving
fig.suptitle(f'NN Results, Varying Amount of Training Data\n\
{hidden_layers:.0f} Hidden Layers, {layer_size:.0f} Neurons Each, "{act_func}" \
Activation Function')
plt.show()

# %% NN Implementation using pytorch
import torch
import torch.nn as nn

# --- Get data ready by turning it into tensors --- #
# Device for data and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input data
X_tdata = X_data[:, 0:idx]
X_tdata = torch.Tensor(X_tdata.T)
X_tdata = X_tdata.to(device)

# Input labels
X_ttruth = X_truth[0:idx]
X_ttruth = torch.Tensor(X_ttruth.T)
X_ttruth = X_ttruth.to(device)

# --- Build a model --- #


class CloudClassr(nn.Module):
    def __init__(self, N, K, layer_size):
        super(CloudClassr, self).__init__()
        # self.flatten = nn.Flatten()     # Flattens data
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
            nn.Linear(layer_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pred = self.linear_ReLU_stack(x)
        return x_pred


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CloudClassr(X_tdata.shape[1], K, layer_size)  # .to(device)X_pred = model(X_tdata)
model.to(device)

# --- Training the model --- #
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10    # number of training loops

# Setting model to training mode
model.train()
for epoch in range(epochs):
    # Forward pass: Predicting the labels on the training data
    X_pred = model(X_tdata)

    # Calculate loss
    loss = criterion(X_pred, X_ttruth)

    # idk what this is
    optimizer.zero_grad() 

    # Backpropagation
    loss.backward()

    # gradient descent)
    optimizer.step()
model.eval()
