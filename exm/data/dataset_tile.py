from nd2reader import ND2Reader
import h5py
import os

class datasetTile:

    def __init__(self):
        self.vol = None
    
    def loadVols(self, vol_path: str, iter_axes: str = 'v', channel: str = None):

        if '.nd2' in vol_path:

            vol = ND2Reader(vol_path)

            if iter_axes:
                vol.iter_axes = iter_axes
    
            self.vol = vol

        elif '.h5' in vol_path:
            self.vol = h5py.File(vol_path, 'r')[channel]
    
        return self.vol

    def setMasks(self, mask_fix, mask_move = None):
        
        self.mask_fix = mask_fix

        if mask_move is not None:
            self.mask_move = mask_move

    def saveH5(self, vol_save, vol_path: str, channel: str):

        if os.path.exists(vol_path):
            with h5py.File(vol_path, 'r+') as f:
                f.create_dataset(channel, vol_save.shape, compression="gzip", dtype=vol_save.dtype, data = vol_save)

        else:
            with h5py.File(vol_path, 'w') as f:
                f.create_dataset(channel, vol_save.shape, compression="gzip", dtype=vol_save.dtype, data = vol_save)

    

    
