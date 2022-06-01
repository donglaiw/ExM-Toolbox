from .io.io import nd2ToVol
import h5py

class datasetTile:

    def __init__(self):
        self.vol_fix = None
        self.vol_move = None
        self.fov = None
    
    def loadVols(self, vol_fix_path: str, vol_mov_path: str, fov: int = None, channel: str = '405 SD'):
        
        if fov:
            self.fov = fov

        if '.nd2' in vol_fix_path:
            self.vol_fix = nd2ToVol(vol_fix_path, fov, channel)
        elif '.h5' in vol_fix_path:
            channel_num = channel[:4]
            self.vol_fix = h5py.File(vol_fix_path, 'r')[channel_num]

        if '.nd2' in vol_mov_path:
            self.vol_move = nd2ToVol(vol_mov_path, fov, channel)
        elif '.h5' in vol_mov_path:
            channel_num = channel[:3]
            self.vol_move = h5py.File(vol_mov_path, 'r')[channel_num]
    
        return self.vol_fix, self.vol_move, self.fov
