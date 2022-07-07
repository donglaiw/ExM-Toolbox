import os
import h5py
import numpy as np
from tifffile import imwrite

def createTIFF(f_vol_out: str, channel_names: list):
    '''
    args: f_vol_out     -> path of the output volume (h5 file)
          channel_names -> channel names in the volume
    returns a numpy image array
    '''
    # make more modular (change needed)
    output = np.zeros((4, 81, 2048, 2048)).astype(np.uint16)

    for i in range(len(channel_names)):
        channel_name = channel_names[i]
        output[i] = np.array(h5py.File(f_vol_out, 'r')[channel_name].astype(np.uint16))

    output = np.swapaxes(output, 0, 1)

    return output


def writeTIFF(img_name: str, work_path: str, np_img: np.ndarray, channel_names: list):
    '''
    args: img_name      -> name of TIFF file
          np_img        -> numpy array from h5 file
          channel_names -> channel names in volume
    writes a TIFF file
    '''
    os.chdir(work_path)
    imwrite(img_name, np_img,
                    metadata={'Channel': {'Name': channel_names}})


def createROI(im_fpath: str, ROI: list):
    '''
    create a ROI within specified coordinate range
    '''
    assert len(ROI) == 4, "(x1, x2, y1, y2) coordinate limits for ROI"

    im = tiffile.imread(im_fpath)
    im_ROI = list()
    for i in range(im.shape[0]):
        channel = im[:,i]
        slice_ROI = list()
        for j in range(channel.shape[0]):
            im_slice = channel[i, :, :,]
            slice_ROI.append(im_slice[ROI[0]:ROI[1], ROI[2]:ROI[3]:,])
        im_ROI.append(np.array(slice_ROI))

    im_ROI = np.array(im_ROI)
    im_ROI = np.swapaxes(im_ROI, 0, 1)

    return np.array(im_ROI)

