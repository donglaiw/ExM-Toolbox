from ..io.io import nd2ToVol
import h5py
import os

class datasetTile:

    def __init__(self):
        self.vol_fix = None
        self.vol_move = None
        self.vol_h5 = None
        self.fov = None
    
    def loadVols(self, fov: int, vol_fix_path: str, vol_mov_path: str, channel: str, result_path: str = None):
        
        if fov:
            self.fov = fov

        if '.nd2' in vol_fix_path:
            self.vol_fix = nd2ToVol(vol_fix_path, self.fov, channel+' SD')
        elif '.h5' in vol_fix_path:
            self.vol_fix = h5py.File(vol_fix_path, 'r')[channel]

        if '.nd2' in vol_mov_path:
            self.vol_move = nd2ToVol(vol_mov_path, self.fov, channel+' SD')
        elif '.h5' in vol_mov_path:
            self.vol_move = h5py.File(vol_mov_path, 'r')[channel]

        if result_path:
            self.vol_h5 = h5py.File(result_path, 'r')[channel]
    
        return self.vol_fix, self.vol_move, self.vol_h5

    def setResult(self, result):
        self.vol_h5 = result

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

    

    
