from nd2reader import ND2Reader
import h5py
import os

class datasetVolume:

    def __init__(self):
        self.vol = None
        self.mask = None
    
    def loadVol(self, vol_path: str, iter_axes: str = 'v', fov: int = None, channel: str = None):

        if '.nd2' in vol_path:

            vol = ND2Reader(vol_path)
            vol.bundle_axes = 'zyx' #nd2 format

            if iter_axes:
                vol.iter_axes = iter_axes
            
            if fov:
                self.vol = vol[fov]
    
            self.vol = vol

        elif '.h5' in vol_path:
            self.vol = h5py.File(vol_path, 'r')[channel]
    
        return self.vol

    def setMasks(self, mask = None):

        if mask is not None:
            self.mask = mask

    def saveH5(self, vol_save, vol_path: str, channel: str):

        if os.path.exists(vol_path):
            with h5py.File(vol_path, 'r+') as f:
                f.create_dataset(channel, vol_save.shape, compression="gzip", dtype=vol_save.dtype, data = vol_save)

        else:
            with h5py.File(vol_path, 'w') as f:
                f.create_dataset(channel, vol_save.shape, compression="gzip", dtype=vol_save.dtype, data = vol_save)

    

    
