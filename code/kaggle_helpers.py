# -*- coding: utf-8 -*-
"""
Helper functions for the project 

Created on Fri Oct 25 13:45:35 2024

@author: J. Monaco
"""

# %% Imports
import numpy as np

# %% Mask helpers


def rle_decode(mask_rle: list[int], shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    Modified from: https://www.kaggle.com/code/artgor/segmentation-in-pytorch-using-convenient-tools#Importing-libraries
    '''
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')
