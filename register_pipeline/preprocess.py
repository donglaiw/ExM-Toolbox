import os
import tiffile
import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image, ImageOps
from tiffile import imsave
from skimage.measure import label
from skimage.color import label2rgb


def denoise(img: np.ndarray, h: int, templateWindowSize=7, searchWindowSize=21):
    '''
    h = 50 is a good starting value
    '''
    assert img.dtype == 'uint8', "image matrix must be of type uint8"
    return cv.fastNLMeansDenoising(img, h, templateWindowSize,
                                    searchWindowSize)


def threshold(img: np.ndarray, thresh_val, maxval):

    (T, threshInv) = cv.threshold(img, thresh=thresh_val, maxval=maxval,
                                   thresh_type=cv.THRESH_BINARY_INV)
    threshInv = 255 - threshInv
    return threshInv


def filter_image(img: np.ndarray, min_idx: int):
    assert min_idx > 1, "minimum area index must be greater than 1"

    upper_bound_idx = []
    lower_bound_idx = []

    labeled_img = label(img, background=0)
    indices, area = np.unique(labeled_img, return_counts=True)
    sorted_area = sorted(area)

    # indices of higher area regions
    for i in range(len(area)):
        for j in range(2, min_idx):
            if (area[j] == sorted_area[len(area) - 1]):
                upper_bound_idx.append(i)

    for i in range(len(area)):
        lower_bound_idx.append(i)

    # indices of low area regions (noise)
    for idx in upper_bound_idx:
        lower_bound_idx.remove(idx)

    # mask out noise
    for idx in lower_bound_idx:
        labeled_img[np.where(labeled_img==idx)] = 0

    return labeled_img

