import os
import tiffile
import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps
from tiffile import imsave
from skimage.measure import label
from skimage.color import label2rgb

#output_dir = '/mp/nas3/Margaret_mouse/'


def retrieve_output(output_dir):
    '''
    args: output_dir -> retrieve TIF files from specified directory
    '''
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for name in files:
            if 'tif' in name:
                print(os.path.join(root, name))
                return tiffile.imread(os.path.join(root, name))


def create_norm_jpg(channel, output_dir):
    '''
    args: channel -> a single channel of the image
    returns: None (stores jpg image of each slice in the
                   specified directory)
    '''
    for i in range(channel.shape[0]):
        vol_slice = channel[i, :, :,]
        # normalize image
        norm_slice = np.clip(vol_slice, np.percentile(vol_slice, 5),
                             np.percentile(vol_slice, 97))
        jpg_img = Image.fromarray(norm_slice)
        jpg_img = jpg_img.convert("L")
        os.chdir(output_dir)
        jpg_img.save("slice_%d.jpg"%i)


def denoise_img(img_loc, h, templateWindowSize=7, searchWindowSize=21):
    '''
    args: img_loc -> filepath of the noisy image
          h       -> parameter regulating filter strength
                     (big h removes noise along with some image details)
          templateWindowSize -> template patch used to compute weights (default is 7)
          searchWindowSize   -> pixel size of window for computing weighted avg of a pixel
    returns -> denoised image
    '''

    noisy_img = cv2.imread(img_loc)
    # convert to grayscale
    noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
    # denoise the image
    return cv2.fastNLMeansDenoising(noisy_img, h, templateWindowSize,
                                    searchWindowSize)


def adaptive_threshold(img, thresh_val, maxval, thresh_type=cv2.THRESH_BINARY_INV):
    '''
    args: img -> image as a numpy array
          thresh_val -> thresholding value
          maxval     -> maximum value to use with binary thresholding
          thresh_type -> type of thresholding used
    returns: image matrix after adaptive thresholding
    '''
    (T, threshInv) = cv2.threshold(img, thresh=thresh_val, maxval=maxval,
                                   thresh_type)
    threshInv = 255 - threshInv
    return threshInv

