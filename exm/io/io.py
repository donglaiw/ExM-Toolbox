import os, sys
import numpy as np
import h5py
import pandas as pd
from nd2reader import ND2Reader
import statistics
from tifffile import imread
from .image import imAdjust
from PIL import Image

def mkdir(fn,opt=0):
    if opt == 1 :# until the last /
        fn = fn[:fn.rfind('/')]
    if not os.path.exists(fn):
        if opt==2:
            os.makedirs(fn)
        else:
            os.mkdir(fn)
# h5 files
def readH5(filename, datasetname=None):
    fid = h5py.File(filename,'r')
    if datasetname is None:
        if sys.version[0]=='2': # py2
            datasetname = fid.keys()
        else: # py3
            datasetname = list(fid)
    if len(datasetname) == 1:
        datasetname = datasetname[0]
    if isinstance(datasetname, (list,)):
        out=[None]*len(datasetname)
        for di,d in enumerate(datasetname):
            out[di] = np.array(fid[d])
        return out
    else:
        return np.array(fid[datasetname])

def writeH5(filename, dtarray, datasetname='main'):
    import h5py
    fid=h5py.File(filename,'w')
    if isinstance(datasetname, (list,)):
        for i,dd in enumerate(datasetname):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()

def readXlsx(xlsx_file, sheet_name='Experiment Data'):
    df = pd.read_excel(
        open(xlsx_file, 'rb'),
        engine='openpyxl',
        header = [1],
        sheet_name=sheet_name)
    # drop invalid rows
    flag = []
    for x in df['Point Name']:
        if isinstance(x, str) and ('#' in x):
            flag.append(False)
        else:
            flag.append(True)
    df = df.drop(df[flag].index)
    flag = []
    for x in df['X Pos[µm]']:
        if isinstance(x, float) or isinstance(x, int):
            flag.append(False)
        else:
            flag.append(True)
    df = df.drop(df[flag].index)
    # select columns
    zz, yy, xx = np.array(df['Z Pos[µm]'].values), np.array(df['Y Pos[µm]'].values), np.array(df['X Pos[µm]'].values)
    ii = np.array([int(x[1:])-1 for x in df['Point Name'].values])
    # need to flip x
    out = np.vstack([zz,yy,-xx,ii]).T.astype(float)
    if (ii==0).sum() != 1:
        loop_ind = np.hstack([np.where(ii==0)[0], len(ii)])
        loop_len = loop_ind[1:] - loop_ind[:-1]
        print('exist %d multipoint loops with length' % len(loop_len), loop_len)
        mid = np.argmax(loop_len)
        out = out[loop_ind[mid]:loop_ind[mid+1]]
        # take the longest one
    return out 


def readNd2(nd2_file, do_info = True):
    vol = ND2Reader(nd2_file)
    info = {}
    if do_info: 
        meta = vol.metadata
        # assume zyx order
        info['tiles_size'] = np.array([meta['z_levels'][-1]+1, meta['height'], meta['width']])
        zz = np.array(meta['z_coordinates'])
        zz_res = statistics.mode(np.round(10000 * (zz[1:]-zz[:-1])) / 10000)
        info['resolution'] = np.array([zz_res, meta['pixel_microns'], meta['pixel_microns']])
        info['channels'] = meta['channels']
    return vol, info

def tiff2H5(tiff_file, h5_file, chunk_size=(100,1024,1024), step=100, im_thres=None):
    # get tiff volume dimension
    img = Image.open(tiff_file)
    num_z = img.n_frames
    test_page = imread(tiff_file, key=range(1))
    sz = [num_z, test_page.shape[0], test_page.shape[1]]

    fid = h5py.File(h5_file, 'w')
    dtype = np.uint8 if im_thres is not None else test_page.dtype
    ds = fid.create_dataset('main', sz, compression="gzip", dtype=dtype, chunks=chunk_size)
    
    num_zi = (sz[0]+step-1) // step
    for zi in range(num_zi):
        z = min((zi+1)*step, sz[0])
        im = imread(tiff_file, key=range(zi*step, z))
        if im_thres is not None:
            im = imAdjust(im, im_thres).astype(np.uint8)
        ds[zi*step:z] = im
    fid.close()
