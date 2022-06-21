#!/usr/bin/env python3

import os
import h5py
from tiffile import imsave


def writeTIFF(data, output_dir, file_name):
    '''
    Write a TIFF file
    args:   data       -> data to be written
            output_dir -> output directory of h5 file
            file_name  -> name of the h5 file
    '''
    os.chdir(output_dir)
    tiffile.imsave(file_name, data)


def writeH5(data, output_dir, file_name, dset_name):
    '''
    Write a h5 file
    args:   data       -> data to be written
            output_dir -> output directory of h5 file
            file_name  -> name of the h5 file
            dset_name  -> name of the dataset
    '''
    os.chdir(output_dir)
    file_obj = h5py.File(file_name, 'w')
    file_obj.create_dataset(dset_name, data=data)
    file_obj.close()


def readH5(path, dset_name):
    '''
    Read a h5 file
    args:   path      -> path to the h5 file
            dset_name -> name of the dataset
    '''
    file_obj = h5py.File(path, 'r')
    return file_obj.get(dset_name)

