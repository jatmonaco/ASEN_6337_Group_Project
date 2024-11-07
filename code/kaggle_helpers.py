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
import pandas as pd
from sklearn.decomposition import PCA


# %% Matrix helpers


def norm_matrix(A: np.array) -> np.array:
    '''
    Take a matrix and normalize values between 0 and 1
    '''
    A = np.array(A)
    return (A - A.min()) / A.ptp()

# %% Helping with data


def DICE_score(img1, img2):
    ''' 
    Calculates DICE score. 

    Taken from: https://www.kaggle.com/code/artgor/segmentation-in-pytorch-using-convenient-tools#Helper-functions-and-classes
    '''
    img1 = np.asarray(img1).astype(bool)
    img2 = np.asarray(img2).astype(bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


class CloudDataset_PCA_1label(Dataset):
    '''
    A pytorch dataloader for the cloud dataset. 
    Based on the function of the same name found here: 
        https://www.kaggle.com/code/dhananjay3/image-segmentation-from-scratch-in-pytorch#Helper-functions

    Returns normalized, single-channel 2D array of PCA'd image, and unmodified mask
    '''

    def __init__(
        self,
        df: pd.DataFrame = None,
        label: str = 'Sugar',
        datatype: str = "train",
        img_paths: str = './understanding_cloud_organization'
    ):
        self.df = df
        self.label = label
        if datatype != "test":
            self.data_folder = f"{img_paths}/train_images"
        else:
            self.data_folder = f"{img_paths}/test_images"
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
    A pytorch dataloader for the cloud dataset. 
    Based on the function of the same name found here: 
        https://www.kaggle.com/code/dhananjay3/image-segmentation-from-scratch-in-pytorch#Helper-functions

    Returns normalized, single-channel 2D array of PCA'd image, and unmodified mask
    '''

    def __init__(
        self,
        df: pd.DataFrame = None,
        datatype: str = "train",
        img_paths: str = './understanding_cloud_organization'
    ):
        self.df = df
        self.labels = ['Sugar', 'Flower', 'Gravel', 'Fish']
        if datatype != "test":
            self.data_folder = f"{img_paths}/train_images"
        else:
            self.data_folder = f"{img_paths}/test_images"
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
            mask = rle2mask(rle)
            masks.append(mask)
        return img_PCA, np.array(masks)

    def __len__(self):
        return len(self.df)


def str2intarr(int_string: str) -> list:
    '''
    Takes in string of form: '[int0, int1, ... intN]' and returns the 
    appropriate np array 
    '''
    arr = np.fromstring(int_string, sep=' ', dtype=int)
    if len(arr) < 1:
        arr = np.nan
    return arr


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
