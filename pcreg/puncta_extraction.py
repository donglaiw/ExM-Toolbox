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

output_dir = '/mp/nas3/Margaret_mouse/'

def retrieve_output(output_dir):
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for name in files:
            if 'tif' in name:
                print(os.path.join(root, name))
                return tiffile.imread(os.path.join(root, name))

img = retrieve_output(output_dir)
channel_2 = img[:, 2]

seg_list = []

for i in range(channel_2.shape[0]):
    vol_slice = channel_2[i,:, :,]
    # normalize image
    norm_slice = np.clip(vol_slice, np.percentile(vol_slice, 5), np.percentile(vol_slice, 97))

    # convert to appropriate format (one time thing)
    #jpg_img = Image.fromarray(norm_slice)
    #jpg_img = jpg_img.convert("L")
    #jpg_img.save("slice_%d.jpg"%i)

    # denoise image + adaptive thresholding

    noisy_img = cv2.imread('slice_%d.jpg'%i)
    noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
    denoised_img = cv2.fastNlMeansDenoising(noisy_img, h=50, templateWindowSize=7, searchWindowSize=21)
    (T, threshInv) = cv2.threshold(denoised_img, 150, 200, cv2.THRESH_BINARY_INV)
    threshInv = 255 - threshInv
    seg_list.append(threshInv)

final_seg = np.stack(seg_list)

