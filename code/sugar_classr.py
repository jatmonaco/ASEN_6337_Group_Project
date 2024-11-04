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
import numpy as np
import pandas as pd
import kaggle_helpers as KH
import torch

# Plotting
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cm'

# %% Loading in the image data
label = 'Sugar'
path = './understanding_cloud_organization'     # path to images
label_keys = pd.read_csv('better_df.csv')       # Mask dataframe in better format
