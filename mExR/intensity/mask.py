import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
from skimage.measure import label
from tiffile import imread, imwrite


def filterImage(img: np.ndarray, minIdx: int) -> np.ndarray:
    '''
    Mask small, noisy elements from an image below
    a certain threshold area
    
    Arguments:
        img:    image volume to be masked
        minIdx: number of small structures to be masked
        
    Returns a filtered image volume
    '''
    upperBound, lowerBound = [], []
    labeled_img = label(img, background=0)
    indices, area = np.unique(labeled_img, return_counts=True) # get area and index of objects
    sorted_area = sorted(area)
    
    # indices of high area regions
    for i in range(len(area)):
        for j in range(2, minIdx):
            if(area[i] == sorted_area[len(area) - j]):
                upperBound.append(i)
    
    # indices of small area objects
    lowerBound = [idx for idx in range(len(area)) if idx not in upperBound]
    
    for idx in upperBound:
        try:
            lowerBound.remove(idx)
        except:
            continue
    
    # mask out noise
    for idx in lowerBound:
        labeled_img[np.where(labeled_img==idx)] = 0
    
    return labeled_img


def roiSlice(z_init: int, n_slices: int, im_channel: np.ndarray, min_idx: int, h_val: int, thresh_val: int, denoise: bool, mask: bool) -> np.ndarray:
    '''
    Mask an entire image volume
    
    Arguments:
        z_init:     initial z-slice of volume
        n_slices:   number of slices to mask
        im_channel: single channel from an image volume
        min_idx:    minimum area index to be masked out
        h_val:      strength of denoising filter
        thresh_val: lower bound of thresholding pixels
        denoise:    check whether or not denoising is needed
        mask:       check whether or not we want to mask small components
    
    Returns a masked image volume
    '''
    assert n_slices <= im_channel.shape[0], 'cannot exceed z-axis limit'
    assert z_init <= im_channel.shape[0], 'cannot exceed z-axis limit'
    assert min_idx > 1, 'minimum area index must be greater than 1'
    
    im_slice_list = list()
    
    for i in tqdm(range(n_slices)):
        im_slice = im_channel[z_init+i, :, :,]
    
        if denoise:
            # denoise + binary thresholding
            denoised = cv.fastNlMeansDenoising(im_slice.astype('uint8'), h=h_val, templateWindowSize=7, searchWindowSize=21)
            (T, threshInv) = cv.threshold(denoised, thresh=thresh_val, maxval=255, type=cv.THRESH_BINARY_INV)
            thresh = 255 - threshInv
       
            # masking of small objects
            im_slice = threshInv
            if mask:
                im_slice = filterImage(threshInv, minIdx=min_idx).astype('uint16')
        
        im_slice_list.append(im_slice)
        
    return np.array(im_slice_list)