# -*- coding: utf-8 -*-
"""
Helper functions for the project 

Created on Fri Oct 25 13:45:35 2024

@@author  J. Monaco
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
    Take a matrix and normalize values between 0 and 1.

    @author  J. Monaco
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


def dice(img1, img2):
    '''
    DICE score between two numpy arrays. 

    Taken from: https://www.kaggle.com/code/artgor/segmentation-in-pytorch-using-convenient-tools#Helper-functions-and-classes
    '''
    img1 = np.asarray(img1).astype(bool)
    img2 = np.asarray(img2).astype(bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())

# %% Datasets and dataloader classes for NN


class CloudDataset_PCA_scaled(Dataset):
    '''
    A pytorch dataloader for the cloud dataset that applies PCA and downscales 
    the image by a factor, downscales and mask by a factor, and pads both 
    to make sure they're a multiple of min_kernel_wd. 

    Returns img and mask as pytorch tensors on the target device. 

    Loosely based on the function of the same name found here: 
        https://www.kaggle.com/code/dhananjay3/image-segmentation-from-scratch-in-pytorch#Helper-functions

    @author  J. Monaco
    '''

    def __init__(
        self,
        df: pd.DataFrame = None,                                                # dataframe of images and labels
        downscale_factor: int = 4,                                              # Factor to downscale the imgs and masks by, in each dimention
        img_paths: str = './understanding_cloud_organization/train_imgs',       # Path where the images are kept
        device: str = 'cuda',                                                   # Device where pytorch performs operations
        min_kernel_wd: int = 4                                                  # The downscaled images dimentionality will be padded to be an integer multiple of this number
    ):
        self.df = df
        self.labels = ['Sugar', 'Flower', 'Gravel', 'Fish']
        self.dscale = int(downscale_factor)
        self.data_folder = img_paths
        self.device = device

        # Getting downsampling dimentions by investigating random image
        rand_img_df = self.df.sample(1).iloc[0]
        image_name = rand_img_df.im_id
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Dimentionality of input image
        ht, wd, n_clrs = img.shape
        self.ht = ht
        self.wd = wd
        self.n_clrs = n_clrs

        # Calculating required padding to fit kernels
        self.ht_pad = min_kernel_wd - (ht // self.dscale % min_kernel_wd)       # Total amount of padding to be added for the ht
        self.wd_pad = min_kernel_wd - (wd // self.dscale % min_kernel_wd)       # Total amount of padding to be added for the wd
        self.wd_pad_L = self.wd_pad // 2        # Left padding
        self.wd_pad_R = -(self.wd_pad // -2)    # Right padding
        self.ht_pad_T = self.ht_pad // 2        # Top padding
        self.ht_pad_B = -(self.ht_pad // -2)    # Bottom padding

    def __getitem__(self, idx):
        # --- Getting image --- #
        img_df = self.df.iloc[idx]
        image_name = img_df.im_id
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- PCA data reduction --- #
        X = img.reshape(-1, self.n_clrs)
        pca = PCA(n_components=1)
        pca.fit(X)
        img_PCA = pca.fit_transform(X)
        img_PCA = np.reshape(img_PCA, (self.ht, self.wd, 1))
        img_PCA = norm_matrix(img_PCA)

        # --- Sending the image to the GPU --- #
        img_PCA = torch.Tensor(img_PCA).float().to(self.device)
        img_PCA = img_PCA.permute(2, 0, 1)                  # Permuting so the number of channels is first dimention

        # --- Downscaling the image and padding it --- #
        img_PCA = F.interpolate(img_PCA.unsqueeze(1),       # An extra dimention is required for the interpolate funciton
                                size=(self.ht // self.dscale, self.wd // self.dscale))
        img_PCA = F.pad(img_PCA,                            # Padding the image to make it compatable with the minimum kernel size
                        pad=(self.wd_pad_L, self.wd_pad_R,
                             self.ht_pad_T, self.ht_pad_B))
        img_PCA = img_PCA.squeeze(1)                        # Taking away extra deimtnion

        # --- Getting mask of this label --- #
        masks = []
        for label in self.labels:
            rle = img_df[f'{label}']
            mask = rle2mask(rle, shape=(self.ht, self.wd))
            masks.append(mask)
        masks = np.array(masks)

        #  --- Sending masks to GPU --- #
        masks = torch.Tensor(masks).float().to(self.device)

        # --- Downscaling the masks and padding it --- #
        masks = F.interpolate(masks.unsqueeze(1),   # An extra dimention is required for the interpolate funciton
                              size=(self.ht // self.dscale, self.wd // self.dscale))
        masks = masks.round()                       # Rounding anti-aliasing to enforce binary masks
        masks = F.pad(masks,                        # Padding the masks to make it compatable with the minimum kernel size
                      pad=(self.wd_pad_L, self.wd_pad_R,
                           self.ht_pad_T, self.ht_pad_B))
        masks = masks.squeeze(1)                    # Taking away extra deimtnion
        return img_PCA, masks

    def __len__(self):
        return len(self.df)
