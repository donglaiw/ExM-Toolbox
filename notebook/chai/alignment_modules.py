import nd2
from nd2reader import ND2Reader
import numpy as np
import napari
import h5py

def nd2ToVol(filename, channel_name='405 SD', ratio=1):
    # volume in zyx order
    vol = ND2Reader(filename)
    channel_names = vol.metadata['channels']
    print('Available channels:', channel_names)
    channel_id = [x for x in range(len(channel_names)) if channel_name in channel_names[x]]
    assert len(channel_id) == 1
    channel_id = channel_id[0]

    out = np.zeros([len(vol), vol[0].shape[0] // ratio, vol[0].shape[1] // ratio], np.uint16)
    for z in range(len(vol)):
        out[z] = vol.get_frame_2D(c=channel_id, t=0, z=z, x=0, y=0, v=0)[::ratio, ::ratio]
    return out

def display_vol(f_vol_fix, f_vol_out, channel_name, ratio = [1,1,1]):
    # ratio: display downsampled volume
    img_fix = nd2ToVol(f_vol_fix, channel_name)
    viewer.add_image(img_fix[::ratio[0], ::ratio[1], ::ratio[2]], \
                      name = 'fixed-'+channel_name, \
                      scale = m_resolution[::-1])

    img_warp = np.array(h5py.File(f_vol_out, 'r')[channel_name])
    viewer.add_image(img_warp[::ratio[0], ::ratio[1], ::ratio[2]], \
                      name = 'warped-'+channel_name, \
                      scale = m_resolution[::-1])
