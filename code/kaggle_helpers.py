# -*- coding: utf-8 -*-
"""
Helper functions for the project 

Created on Fri Oct 25 13:45:35 2024

@author: J. Monaco
"""

# %% Imports
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.decomposition import PCA
from skimage.transform import resize
import torch.nn.functional as F


# %% Matrix helpers


def norm_matrix(A: np.array) -> np.array:
    '''
    Take a matrix and normalize values between 0 and 1
    '''
    A = np.array(A)
    return (A - A.min()) / np.ptp(A)


def str2intarr(int_string: str) -> list:
    '''
    Takes in string of form: '[int0, int1, ... intN]' and returns the 
    appropriate np array 
    '''
    arr = np.fromstring(int_string, sep=' ', dtype=int)
    if len(arr) < 1:
        arr = np.nan
    return arr
# %% Helping with data


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated

    Taken from: https://www.kaggle.com/code/artgor/segmentation-in-pytorch-using-convenient-tools#Helper-functions-and-classes
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle: list[int], shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    Modified from: https://www.kaggle.com/code/artgor/segmentation-in-pytorch-using-convenient-tools#Importing-libraries
    '''
    # Return all zeros if no mask
    if np.isnan(mask_rle).any():
        return np.zeros(shape)

    # Converting run-line encoding to 2D array
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    # img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    img = np.zeros(shape[0] * shape[1], dtype=float)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def get_img(img_name: str,
            kaggle_dir: str = './understanding_cloud_organization/train_images',
            img_dir: str = '/train_images'):
    """
    Return image based on image name and folder.
    adapted from: https://www.kaggle.com/code/artgor/segmentation-in-pytorch-using-convenient-tools#Helper-functions-and-classes
    """
    img_dir = kaggle_dir + img_dir
    image_path = os.path.join(img_dir, img_name)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img, dtype=float)

# %% Datasets and dataloader classes for NN


class CloudDataset_PCA_1label(Dataset):
    '''
    A pytorch dataloader for the cloud dataset, returning just 1 class
    Based on the class found here: 
        https://www.kaggle.com/code/dhananjay3/image-segmentation-from-scratch-in-pytorch#Helper-functions

    Returns normalized, single-channel 2D array of PCA'd image, and unmodified mask
    '''

    def __init__(
        self,
        df: pd.DataFrame = None,
        label: str = 'Sugar',
        img_paths: str = './understanding_cloud_organization'
    ):
        self.df = df
        self.label = label
        self.data_folder = img_paths
        self.labels = ['Sugar', 'Flower', 'Gravel', 'Fish']

    def __getitem__(self, idx):
        img_df = self.df.iloc[idx]

        # Getting image
        image_name = img_df.im_id
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # PCA data reduction
        ht, wd, n_clrs = img.shape
        X = img.reshape(-1, n_clrs)
        pca = PCA(n_components=1)
        pca.fit(X)
        img_PCA = pca.fit_transform(X)
        img_PCA = np.reshape(img_PCA, (ht, wd, 1))
        img_PCA = norm_matrix(img_PCA)

        # Getting mask of this label
        rle = img_df[f'{self.label}']
        mask = rle2mask(rle)
        return img_PCA, mask

    def __len__(self):
        return len(self.df)


class CloudDataset_PCA(Dataset):
    '''
    A pytorch dataloader for the cloud dataset, returning the 1D PCA'd img and its 4D class masks 
    Based on the function found here: 
        https://www.kaggle.com/code/dhananjay3/image-segmentation-from-scratch-in-pytorch#Helper-functions

    Returns normalized, single-channel 2D array of PCA'd image, and unmodified mask
    '''

    def __init__(
        self,
        df: pd.DataFrame = None,
        img_paths: str = './understanding_cloud_organization/train_imgs'
    ):
        self.df = df
        self.labels = ['Sugar', 'Flower', 'Gravel', 'Fish']
        self.data_folder = img_paths
        self.labels = ['Sugar', 'Flower', 'Gravel', 'Fish']

    def __getitem__(self, idx):
        img_df = self.df.iloc[idx]

        # Getting image
        image_name = img_df.im_id
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # PCA data reduction
        ht, wd, n_clrs = img.shape
        X = img.reshape(-1, n_clrs)
        pca = PCA(n_components=1)
        pca.fit(X)
        img_PCA = pca.fit_transform(X)
        img_PCA = np.reshape(img_PCA, (ht, wd, 1))
        img_PCA = norm_matrix(img_PCA)

        # Getting mask of this label
        masks = []
        for label in self.labels:
            rle = img_df[f'{label}']
            mask = rle2mask(rle, shape=(ht, wd))
            masks.append(mask)
        return img_PCA, np.array(masks)

    def __len__(self):
        return len(self.df)


class CloudDataset_PCA_scaled(Dataset):
    '''
    A pytorch dataloader for the cloud dataset that downscales the image and 
    mask by a factor 

    Based on the function of the same name found here: 
        https://www.kaggle.com/code/dhananjay3/image-segmentation-from-scratch-in-pytorch#Helper-functions

    Returns normalized, single-channel 2D array of PCA'd image, and unmodified mask
    '''

    def __init__(
        self,
        df: pd.DataFrame = None,
        downscale_factor: int = 4,
        img_paths: str = './understanding_cloud_organization/train_imgs'
    ):
        self.df = df
        self.labels = ['Sugar', 'Flower', 'Gravel', 'Fish']
        self.dscale = int(downscale_factor)

        self.data_folder = img_paths
        self.labels = ['Sugar', 'Flower', 'Gravel', 'Fish']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Getting downsampling dimentions by investigating random image
        rand_img_df = self.df.sample(1).iloc[0]
        image_name = rand_img_df.im_id
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Getting dimentionality of images and padding
        ht, wd, n_clrs = img.shape
        self.ht = ht
        self.wd = wd
        self.n_clrs = n_clrs
        self.ht_pad = 4 - (ht // self.dscale % 4)
        self.wd_pad = 4 - (wd // self.dscale % 4)
        self.wd_pad_L = self.wd_pad // 2
        self.wd_pad_R = -(self.wd_pad // -2)
        self.ht_pad_T = self.ht_pad // 2
        self.ht_pad_B = -(self.ht_pad // -2)

    def __getitem__(self, idx):
        img_df = self.df.iloc[idx]

        # Getting image
        image_name = img_df.im_id
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # PCA data reduction
        X = img.reshape(-1, self.n_clrs)
        pca = PCA(n_components=1)
        pca.fit(X)
        img_PCA = pca.fit_transform(X)
        img_PCA = np.reshape(img_PCA, (self.ht, self.wd, 1))
        img_PCA = norm_matrix(img_PCA)

        # Sending the image to the GPU for downsizing
        img_PCA = torch.Tensor(img_PCA).float().to(self.device)
        img_PCA = img_PCA.permute(2, 0, 1)

        # Downscaling the image and making sure it fits the conv. kernels
        img_PCA = F.interpolate(img_PCA.unsqueeze(1),
                                size=(self.ht // self.dscale, self.wd // self.dscale))
        # Padding the image to make it compatable with the minimum kernel size
        img_PCA = F.pad(img_PCA,
                        pad=(self.wd_pad_L, self.wd_pad_R,
                             self.ht_pad_T, self.ht_pad_B))
        img_PCA = img_PCA.squeeze(1)

        # Getting mask of this label
        masks = []
        for label in self.labels:
            rle = img_df[f'{label}']
            mask = rle2mask(rle, shape=(self.ht, self.wd))
            masks.append(mask)
        masks = np.array(masks)

        # Sending masks to GPU
        masks = torch.Tensor(masks).float().to(self.device)

        # Resizing
        masks = F.interpolate(masks.unsqueeze(1),
                              size=(self.ht // self.dscale, self.wd // self.dscale))
        # Padding the masks to make it compatable with the minimum kernel size
        masks = F.pad(masks,
                      pad=(self.wd_pad_L, self.wd_pad_R,
                           self.ht_pad_T, self.ht_pad_B))
        masks = masks.round()
        masks = masks.squeeze(1)
        return img_PCA, masks

    def __len__(self):
        return len(self.df)
